"""
Module for capturing system audio via WASAPI loopback.
"""

import sys
import os
import threading
import time
import numpy as np
import pyaudio
from typing import Optional

# Optional import of sounddevice for WASAPI loopback
try:
    import sounddevice as sd  # type: ignore
    HAVE_SD = True
except Exception:
    sd = None  # type: ignore
    HAVE_SD = False

# Optional import of PyAudioWPatch (WASAPI loopback)
try:
    import pyaudiowpatch as pyaudio_wasapi  # type: ignore
    HAVE_PAW = True
except Exception:
    pyaudio_wasapi = None  # type: ignore
    HAVE_PAW = False


class SystemAudioListener:
    """
    Class for capturing system audio via WASAPI loopback.
    Uses PyAudio to capture system audio.
    """
    
    def __init__(self, samplerate: int = 16000, channels: int = 1, use_wasapi_loopback: bool = False, output_device: Optional[str] = None, input_device_index: Optional[int] = None):
        """
        Initialize system audio listener.
        
        Args:
            samplerate: Sample rate (default 16 kHz)
            channels: Number of channels (default 1 - mono)
        """
        self.samplerate = samplerate
        self.channels = channels
        self.frame_duration_ms = 100  # Frame duration in milliseconds
        self.frame_size = int(samplerate * self.frame_duration_ms / 1000)
        
        # Buffers for audio data
        self.live_buffer = []  # For live transcription
        self.since_enter_buffer = []  # For sending on Enter
        
        # Thread-safety
        self.lock = threading.Lock()
        
        # State
        self.is_recording = False
        self.audio = None
        self.stream = None  # PyAudio/PyAudioWPatch or sounddevice stream
        self.backend = "pyaudio"  # "pyaudio" | "sounddevice"
        self.pa_module = pyaudio  # active PyAudio module (standard or patched)
        self._pa_continue = pyaudio.paContinue
        self._pa_abort = pyaudio.paAbort
        self.stream_channels = channels  # actual number of channels in opened stream

        # Loopback settings
        self.use_wasapi_loopback = bool(use_wasapi_loopback)
        self.output_device_name = output_device
        self.input_device_index = input_device_index
        self.recording_thread = None
        
        # Initialize PyAudio (needed even if we try sounddevice, to keep backward compatibility)
        try:
            self.audio = pyaudio.PyAudio()
            print(f"[info] PyAudio initialized", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyAudio: {e}")
    
    def start_recording(self):
        """Start recording system audio."""
        if self.is_recording:
            return
        
        try:
            # Prefer WASAPI loopback on Windows
            if self.use_wasapi_loopback and sys.platform == 'win32':
                # 1) PyAudioWPatch (reliable option)
                if HAVE_PAW:
                    self._start_recording_pyaudio_wasapi_loopback()
                    return
                # 2) sounddevice (if available)
                if HAVE_SD:
                    self._start_recording_sounddevice_loopback()
                    return

            # Otherwise use the previous path via PyAudio (microphone or loopback device in input list)
            loopback_device = None
            sel_device_info = None

            desired_out_name_lower = None
            if self.use_wasapi_loopback:
                try:
                    if self.output_device_name:
                        desired_out_name_lower = self.output_device_name.lower()
                    else:
                        # By default listen to the default output device
                        out_info = self.audio.get_default_output_device_info()
                        desired_out_name_lower = str(out_info.get('name', '')).lower()
                        print(f"[info] Target default output device: {out_info.get('name', '')}", file=sys.stderr)
                except Exception:
                    desired_out_name_lower = None

            first_loopback_idx = None
            first_loopback_info = None

            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                device_name = str(device_info.get('name', '')).lower()
                # Look for input devices marked as loopback
                if ('loopback' in device_name or 
                    'stereo mix' in device_name or 
                    'what u hear' in device_name or
                    'wave out mix' in device_name):
                    if first_loopback_idx is None:
                        first_loopback_idx = i
                        first_loopback_info = device_info
                    if desired_out_name_lower and desired_out_name_lower in device_name:
                        loopback_device = i
                        sel_device_info = device_info
                        print(f"[info] Found loopback device: {device_info['name']}", file=sys.stderr)
                        break

            if loopback_device is None and first_loopback_idx is not None:
                loopback_device = first_loopback_idx
                sel_device_info = first_loopback_info
                print(f"[info] Found loopback device: {sel_device_info['name']}", file=sys.stderr)

            if loopback_device is None:
                sel_device_info = self.audio.get_default_input_device_info()
                loopback_device = sel_device_info['index']
                print(f"[info] Using default input device: {self.audio.get_device_info_by_index(loopback_device)['name']}", file=sys.stderr)

            # If user explicitly provided input device index — use it
            if self.input_device_index is not None:
                try:
                    sel_device_info = self.audio.get_device_info_by_index(self.input_device_index)
                    loopback_device = self.input_device_index
                    print(f"[info] Using specified device index: {sel_device_info['name']} (#{self.input_device_index})", file=sys.stderr)
                except Exception as e:
                    if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                        print(f"[warning] Provided input device index is unavailable: {e}. Ignoring.", file=sys.stderr)

            # Adjust sample rate and channels to the device
            try:
                dev_sr = int(sel_device_info.get('defaultSampleRate', self.samplerate))  # type: ignore
            except Exception:
                dev_sr = self.samplerate
            self.samplerate = dev_sr
            max_in = int(sel_device_info.get('maxInputChannels', self.channels))  # type: ignore
            self.stream_channels = max(1, min(self.channels, max_in) if max_in > 0 else self.channels)

            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.stream_channels,
                rate=self.samplerate,
                input=True,
                input_device_index=loopback_device,
                frames_per_buffer=self.frame_size,
                stream_callback=self._audio_callback
            )

            self.backend = "pyaudio"
            self.pa_module = pyaudio
            self._pa_continue = pyaudio.paContinue
            self._pa_abort = pyaudio.paAbort
            self.is_recording = True
            self.stream.start_stream()
            print("[info] System audio recording started", file=sys.stderr)

        except Exception as e:
            raise RuntimeError(f"Failed to start recording: {e}")

    def _start_recording_sounddevice_loopback(self):
        """Start recording via sounddevice in WASAPI loopback mode."""
        assert HAVE_SD
        try:
            # Determine output device
            selected_output_index = None
            selected_output_name = None

            devices = sd.query_devices()  # type: ignore
            default_out = None
            try:
                default_pair = sd.default.device  # type: ignore
                if isinstance(default_pair, (list, tuple)) and len(default_pair) >= 2:
                    default_out = default_pair[1]
            except Exception:
                default_out = None

            if self.output_device_name:
                name_lower = self.output_device_name.lower()
                for idx, info in enumerate(devices):
                    if info.get('max_output_channels', 0) > 0:
                        dev_name = str(info.get('name', ''))
                        if name_lower in dev_name.lower():
                            selected_output_index = idx
                            selected_output_name = dev_name
                            break

            if selected_output_index is None:
                if isinstance(default_out, int):
                    selected_output_index = default_out
                    selected_output_name = devices[default_out]['name']
                else:
                    # Find the first available output device
                    for idx, info in enumerate(devices):
                        if info.get('max_output_channels', 0) > 0:
                            selected_output_index = idx
                            selected_output_name = info.get('name', str(idx))
                            break
                if selected_output_index is None:
                    raise RuntimeError("No available output device found for loopback")

            # Create an input stream with loopback=True on the output device
            # Guard older sounddevice versions which may not support the signature
            wasapi_settings = None
            try:
                wasapi_settings = sd.WasapiSettings(loopback=True)  # type: ignore
            except TypeError as e:
                # Try alternative: create and set attribute if available
                try:
                    tmp_settings = sd.WasapiSettings()  # type: ignore
                    try:
                        setattr(tmp_settings, "loopback", True)
                        wasapi_settings = tmp_settings
                        if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                            print(
                                f"[warning] sounddevice.WasapiSettings(loopback=True) unsupported on this version ({getattr(sd, '__version__', 'unknown')}); used attribute-based loopback instead",
                                file=sys.stderr,
                            )
                    except Exception:
                        raise
                except Exception as e2:
                    sd_ver = getattr(sd, "__version__", "unknown")
                    if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                        print(
                            f"[warning] sounddevice WasapiSettings not supported (version {sd_ver}). {type(e).__name__}: {e}. Falling back to PyAudio.",
                            file=sys.stderr,
                        )
                    print(
                        "[info] Falling back to standard PyAudio loopback path (expected on older libs)",
                        file=sys.stderr,
                    )
                    # Fallback to PyAudio
                    self.use_wasapi_loopback = False
                    self._safe_close_sd_stream()
                    self.start_recording()
                    return

            # For stability use 2 channels and then mix down to mono if needed
            sd_channels = max(1, self.channels)

            self.stream = sd.InputStream(  # type: ignore
                samplerate=self.samplerate,
                channels=sd_channels,
                dtype='float32',
                device=selected_output_index,
                blocksize=self.frame_size,
                callback=self._sd_callback,
                extra_settings=wasapi_settings
            )

            self.stream.start()
            self.backend = "sounddevice"
            self.is_recording = True
            print(f"[info] WASAPI loopback enabled. Output device: {selected_output_name}", file=sys.stderr)

        except Exception as e:
            sd_ver = getattr(sd, "__version__", "unknown")
            if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                print(
                    f"[warning] Failed to start WASAPI loopback (sounddevice {sd_ver}): {e}.",
                    file=sys.stderr,
                )
            print(
                "[info] Falling back to standard PyAudio loopback path (expected on older libs)",
                file=sys.stderr,
            )
            # Fallback to PyAudio
            self.use_wasapi_loopback = False
            self._safe_close_sd_stream()
            # Retry via PyAudio
            self.start_recording()

    def _start_recording_pyaudio_wasapi_loopback(self):
        """Start recording via PyAudioWPatch (WASAPI loopback)."""
        assert HAVE_PAW
        try:
            p = pyaudio_wasapi.PyAudio()  # type: ignore
            # Save as the active PyAudio
            if self.audio is not None:
                try:
                    self.audio.terminate()
                except Exception:
                    pass
            self.audio = p

            # Find WASAPI host API
            wasapi_info = p.get_host_api_info_by_type(pyaudio_wasapi.paWASAPI)  # type: ignore
            wasapi_index = wasapi_info.get('index', None)
            if wasapi_index is None:
                raise RuntimeError("WASAPI host API is unavailable")

            selected_output_index = None
            selected_output_name = None

            # If an output device name is provided, search among WASAPI output devices
            if self.output_device_name:
                name_lower = self.output_device_name.lower()
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get('hostApi') == wasapi_index and dev.get('maxOutputChannels', 0) > 0:
                        dev_name = str(dev.get('name', ''))
                        if name_lower in dev_name.lower():
                            selected_output_index = dev['index']
                            selected_output_name = dev_name
                            break

            # Otherwise take the default output device for WASAPI
            if selected_output_index is None:
                default_out = wasapi_info.get('defaultOutputDevice', -1)
                if isinstance(default_out, int) and default_out >= 0:
                    dev = p.get_device_info_by_index(default_out)
                    selected_output_index = dev['index']
                    selected_output_name = dev.get('name', str(default_out))
                else:
                    # Fallback: first available output device in WASAPI
                    for i in range(p.get_device_count()):
                        dev = p.get_device_info_by_index(i)
                        if dev.get('hostApi') == wasapi_index and dev.get('maxOutputChannels', 0) > 0:
                            selected_output_index = dev['index']
                            selected_output_name = dev.get('name', str(i))
                            break

            if selected_output_index is None:
                raise RuntimeError("No WASAPI output device found for loopback")

            # Default sample rate for the selected device
            dev_info = p.get_device_info_by_index(selected_output_index)
            try:
                dev_sr = int(dev_info.get('defaultSampleRate', self.samplerate))  # type: ignore
            except Exception:
                dev_sr = self.samplerate
            self.samplerate = dev_sr

            # For reliability open 2 channels, then mix down to mono if needed
            self.stream_channels = 2 if self.channels == 1 else max(1, self.channels)

            # Open an input stream as loopback from the selected output device (via StreamInfo)
            # Guard against older pyaudiowpatch versions missing WASAPI helpers
            if not hasattr(pyaudio_wasapi, "PaWasapiStreamInfo") or not hasattr(pyaudio_wasapi, "paWinWasapiLoopback"):
                paw_ver = getattr(pyaudio_wasapi, "__version__", "unknown")
                if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                    print(
                        f"[warning] PyAudioWPatch missing PaWasapiStreamInfo (version {paw_ver}). Please upgrade pyaudiowpatch>=0.2.12.",
                        file=sys.stderr,
                    )
                # Try sounddevice path if available
                if HAVE_SD:
                    try:
                        self._start_recording_sounddevice_loopback()
                        return
                    except Exception:
                        pass
                # Fallback to PyAudio
                print(
                    "[info] Falling back to standard PyAudio loopback path (expected on older libs)",
                    file=sys.stderr,
                )
                self.use_wasapi_loopback = False
                self.start_recording()
                return

            wasapi_info = pyaudio_wasapi.PaWasapiStreamInfo(  # type: ignore
                flags=pyaudio_wasapi.paWinWasapiLoopback  # type: ignore
            )

            self.stream = p.open(
                format=pyaudio_wasapi.paFloat32,  # type: ignore
                channels=self.stream_channels,
                rate=self.samplerate,
                input=True,
                input_device_index=selected_output_index,
                frames_per_buffer=self.frame_size,
                stream_callback=self._audio_callback,
                input_host_api_specific_stream_info=wasapi_info
            )

            self.backend = "pyaudio"
            self.pa_module = pyaudio_wasapi  # type: ignore
            self._pa_continue = pyaudio_wasapi.paContinue  # type: ignore
            self._pa_abort = pyaudio_wasapi.paAbort  # type: ignore
            self.is_recording = True
            self.stream.start_stream()
            print(f"[info] WASAPI loopback enabled (PyAudioWPatch). Output device: {selected_output_name}", file=sys.stderr)
            print(f"[info] Effective rate: {self.samplerate} Hz, channels: {self.stream_channels}", file=sys.stderr)

        except Exception as e:
            paw_ver = getattr(pyaudio_wasapi, "__version__", "unknown")
            if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                print(
                    f"[warning] Failed to start WASAPI loopback (PyAudioWPatch {paw_ver}): {e}. Trying fallback modes.",
                    file=sys.stderr,
                )
            # Try sounddevice
            if HAVE_SD:
                try:
                    self._start_recording_sounddevice_loopback()
                    return
                except Exception:
                    pass
            # Otherwise fallback to regular PyAudio
            self.use_wasapi_loopback = False
            print(
                "[info] Falling back to standard PyAudio loopback path (expected on older libs)",
                file=sys.stderr,
            )
            self.start_recording()
    
    def stop_recording(self):
        """Stop recording system audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            if self.backend == "pyaudio":
                self.stream.stop_stream()
                self.stream.close()
            elif self.backend == "sounddevice":
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
            self.stream = None
        
        print("[info] System audio recording stopped", file=sys.stderr)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for processing audio data."""
        try:
            if status:
                if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                    print(f"[warning] Audio stream status: {status}", file=sys.stderr)
            
            if in_data:
                # Convert bytes to numpy float32 array
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # If stereo, convert to mono
                if self.channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                with self.lock:
                    # Append to both buffers
                    self.live_buffer.append(audio_data.copy())
                    self.since_enter_buffer.append(audio_data.copy())
                    
                    # Limit buffer sizes (keep last 30 seconds)
                    max_frames = int(30 * self.samplerate / self.frame_size)
                    if len(self.live_buffer) > max_frames:
                        self.live_buffer = self.live_buffer[-max_frames:]
                    if len(self.since_enter_buffer) > max_frames:
                        self.since_enter_buffer = self.since_enter_buffer[-max_frames:]
            
            return (None, self._pa_continue)
            
        except Exception as e:
            print(f"[error] Error in audio callback: {e}", file=sys.stderr)
            return (None, self._pa_abort)

    def _sd_callback(self, indata, frames, time_info, status):  # type: ignore
        """Callback for sounddevice (WASAPI loopback)."""
        try:
            if status:
                if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
                    print(f"[warning] Audio stream status (sd): {status}", file=sys.stderr)

            if indata is not None:
                audio_data = np.array(indata, dtype=np.float32, copy=False)
                # Normalize to (N,) float32
                if audio_data.ndim == 2:
                    if self.channels == 1:
                        audio_data = audio_data.mean(axis=1)
                    else:
                        # Leave as is, then flatten to 1D
                        audio_data = audio_data.reshape(-1)
                else:
                    audio_data = audio_data.reshape(-1)

                with self.lock:
                    self.live_buffer.append(audio_data.copy())
                    self.since_enter_buffer.append(audio_data.copy())

                    max_frames = int(30 * self.samplerate / self.frame_size)
                    if len(self.live_buffer) > max_frames:
                        self.live_buffer = self.live_buffer[-max_frames:]
                    if len(self.since_enter_buffer) > max_frames:
                        self.since_enter_buffer = self.since_enter_buffer[-max_frames:]
        except Exception as e:
            print(f"[error] Error in sd callback: {e}", file=sys.stderr)

    def _safe_close_sd_stream(self):
        try:
            if self.stream is not None and self.backend == "sounddevice":
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
    
    def get_chunk_and_clear(self) -> np.ndarray:
        """
        Get accumulated data for live transcription and clear the buffer.
        
        Returns:
            numpy.ndarray: Audio data in float32
        """
        with self.lock:
            if not self.live_buffer:
                return np.array([], dtype=np.float32)
            
            # Concatenate all frames into one array
            audio_chunk = np.concatenate(self.live_buffer, axis=0)
            
            # Clear the buffer
            self.live_buffer.clear()
            
            return audio_chunk
    
    def get_since_last_enter_and_clear(self) -> np.ndarray:
        """
        Get accumulated data since last Enter and clear the buffer.
        
        Returns:
            numpy.ndarray: Audio data in float32
        """
        with self.lock:
            if not self.since_enter_buffer:
                return np.array([], dtype=np.float32)
            
            # Concatenate all frames into one array
            audio_chunk = np.concatenate(self.since_enter_buffer, axis=0)
            
            # Clear the buffer
            self.since_enter_buffer.clear()
            
            return audio_chunk
    
    def __enter__(self):
        """Context manager - enter."""
        self.start_recording()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager - exit."""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()


