#!/usr/bin/env python3
"""
System Audio Copilot - CLI tool for live transcription of system audio
with the ability to get hints from an AI assistant.
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional
from datetime import datetime

import numpy as np
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import ctypes
import signal

from audio_capture import SystemAudioListener, MicrophoneListener
from stt import transcribe_audio_chunk
from llm import make_hint, DEFAULT_SYSTEM_PROMPT, DEFAULT_SUMMARY_SYSTEM_PROMPT, make_summary


def load_config():
    """Load configuration from .env located next to the executable/script."""
    # Determine the directory to search for .env
    base_dir = None
    try:
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()

    # First try .env next to the exe/script, then do a standard search
    env_path = os.path.join(base_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv(find_dotenv())

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY not found in environment variables", file=sys.stderr)
        sys.exit(1)
    
    return {
        "api_key": api_key,
        "model_transcribe": os.getenv("MODEL_TRANSCRIBE", "whisper-1"),
        "model_hints": os.getenv("MODEL_HINTS", "gpt-4o-mini"),
        "temperature": float(os.getenv("TEMPERATURE", "0.2")),
        "dev_logs": str(os.getenv("DEV_LOGS", "0")).strip().lower() in ("1", "true", "yes", "on"),
        "assistant_system_prompt": os.getenv("ASSISTANT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        "assistant_summary_prompt": os.getenv("ASSISTANT_SUMMARY_PROMPT", DEFAULT_SUMMARY_SYSTEM_PROMPT),
        "ui_opacity": int(os.getenv("UI_OPACITY", "0") or "0"),
    }


def setup_openai_client(api_key: str) -> OpenAI:
    """Create an OpenAI client."""
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[error] Failed to create OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)


def live_transcription_worker(
    audio_listener: SystemAudioListener,
    client: OpenAI,
    config: dict,
    window_sec: float,
    enter_only: bool,
    vad_threshold: float,
    vad_frame_ms: int,
    vad_min_voiced_ratio: float,
    since_enter_text: list,
    since_enter_lock: threading.Lock,
    session_text: list,
    source_label: str
):
    """
    Background worker for live transcription.
    
    Args:
        audio_listener: Audio listener instance
        client: OpenAI client
        config: Configuration dict
        window_sec: Transcription interval in seconds
        enter_only: Flag for "Enter-only" mode
        since_enter_text: List to accumulate text
        since_enter_lock: Lock for since_enter_text
    """
    while True:
        try:
            time.sleep(window_sec)
            
            # Get audio chunk
            audio_chunk = audio_listener.get_chunk_and_clear()
            
            if len(audio_chunk) > 0:
                # VAD gating over sub-frames
                try:
                    sr = int(audio_listener.samplerate)
                    frame_size = int(max(1, round((vad_frame_ms / 1000.0) * sr)))
                    num_frames = int(np.ceil(len(audio_chunk) / frame_size)) if frame_size > 0 else 1
                    pad_len = num_frames * frame_size - len(audio_chunk)
                    if pad_len > 0:
                        padded = np.pad(audio_chunk, (0, pad_len), mode='constant')
                    else:
                        padded = audio_chunk
                    frames = padded.reshape(num_frames, frame_size)
                    rms_per_frame = np.sqrt(np.mean(np.square(frames), axis=1))
                    voiced_flags = rms_per_frame >= vad_threshold
                    voiced_ratio = float(np.mean(voiced_flags)) if num_frames > 0 else 0.0
                except Exception:
                    voiced_ratio = 0.0

                if voiced_ratio < vad_min_voiced_ratio:
                    # Skip silent/low-voicing windows entirely
                    continue
                # Transcribe
                transcribed_text = transcribe_audio_chunk(
                    audio_chunk, 
                    audio_listener.samplerate, 
                    client, 
                    config["model_transcribe"]
                )
                
                if transcribed_text:
                    # Add to buffer for Enter
                    with since_enter_lock:
                        since_enter_text.append(transcribed_text)
                        # Also accumulate full session transcript with timestamp and label
                        try:
                            ts_label = datetime.now().strftime('%H:%M:%S')
                        except Exception:
                            ts_label = "" 
                        session_text.append((ts_label, source_label, transcribed_text))
                    
                    # Print live transcription if not in enter_only mode
                    if not enter_only:
                        use_color = bool(config.get("use_color", False))
                        colors = config.get("colors", {}) if isinstance(config.get("colors", {}), dict) else {}
                        if source_label == "system":
                            prefix = "[Other]"
                            if use_color:
                                c = colors.get("other", "")
                                r = colors.get("reset", "")
                                prefix = f"{c}[Other]{r}"
                        else:
                            prefix = "[Me]"
                            if use_color:
                                c = colors.get("me", "")
                                r = colors.get("reset", "")
                                prefix = f"{c}[Me]{r}"
                        print(f"{prefix} {transcribed_text}")
                        sys.stdout.flush()
                        
        except Exception as e:
            print(f"[error] Error in transcription thread: {e}", file=sys.stderr)


def handle_enter_input(
    since_enter_text: list,
    since_enter_lock: threading.Lock,
    client: OpenAI,
    config: dict,
    session_hints: list
):
    """
    Handle Enter key press to get an AI hint.
    
    Args:
        since_enter_text: List of accumulated text
        since_enter_lock: Lock for since_enter_text
        client: OpenAI client
        config: Configuration dict
    """
    with since_enter_lock:
        if not since_enter_text:
            if config.get("dev_logs"):
                print("[warning] No accumulated text to send")
            return
        
        # Send only the last up to 3 recognized blocks
        tail_blocks = since_enter_text[-3:] if len(since_enter_text) > 3 else list(since_enter_text)
        full_text = " ".join(tail_blocks).strip()
        
        # Clear the buffer
        since_enter_text.clear()
    
    if not full_text:
        if config.get("dev_logs"):
            print("[warning] No accumulated text to send")
        return
    
    print(f"[sending] Sending text: {full_text}")
    
    # Get a hint from AI
    hint = make_hint(
        full_text,
        client,
        config["model_hints"],
        config["temperature"],
        system_prompt=config.get("assistant_system_prompt")
    )
    
    if hint:
        use_color = bool(config.get("use_color", False))
        colors = config.get("colors", {}) if isinstance(config.get("colors", {}), dict) else {}
        if use_color:
            c = colors.get("assistant", "")
            r = colors.get("reset", "")
            print(f"\n{c}=== ASSISTANT ==={r}")
            print(f"{c}{hint}{r}")
            print(f"{c}{'=' * 20}{r}")
        else:
            print(f"\n=== ASSISTANT ===")
            print(hint)
            print("=" * 20)
        try:
            session_hints.append((full_text, hint))
        except Exception:
            pass
    else:
        print("[error] Failed to get assistant response")
def prompt_and_save_session(
    system_listener: Optional[SystemAudioListener],
    mic_listener: Optional[MicrophoneListener],
    session_text: list,
    session_hints: list,
    save_on_exit_mode: str,
    save_dir_opt: str,
    save_audio_seconds: int,
    save_audio_mode: str,
    mix_gain_mode: str,
    client: Optional[OpenAI],
    config: Optional[dict],
    mix_target_rms_db: float,
    mix_max_gain_db: float,
    mix_min_gain_db: float,
    mix_rms_gate_db: float,
    session_start_ts: float
) -> None:
    """
    Prompt user and save session (audio + text) depending on mode.

    save_on_exit_mode: 'ask' | 'yes' | 'no'
    save_dir_opt: base sessions directory (may be relative)
    save_audio_seconds: how many recent seconds of audio to save
    """
    mode = (save_on_exit_mode or "ask").strip().lower()
    if mode not in ("ask", "yes", "no"):
        mode = "ask"

    should_save = False
    if mode == "yes":
        should_save = True
    elif mode == "no":
        should_save = False
    else:
        # ask
        try:
            resp = input("Сохранить сессию (аудио + текст)? [Y/n] ").strip()
            if resp == "" or resp.lower().startswith("y"):
                should_save = True
            else:
                should_save = False
        except EOFError:
            # Non-interactive: default to no when ask
            print("[warning] Input unavailable, skipping save (use --save-on-exit yes to auto-save)", file=sys.stderr)
            should_save = False
        except KeyboardInterrupt:
            print("\n[info] Cancelled save prompt", file=sys.stderr)
            should_save = False

    if not should_save:
        return

    # Determine base dir for relative save_dir
    base_dir = None
    try:
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.getcwd()
    except Exception:
        base_dir = os.getcwd()

    save_dir_cfg = save_dir_opt or "./sessions"
    if os.path.isabs(save_dir_cfg):
        sessions_root = save_dir_cfg
    else:
        sessions_root = os.path.join(base_dir, save_dir_cfg)

    # Create timestamped session folder
    now_ts = time.time()
    start_dt = datetime.fromtimestamp(session_start_ts)
    end_dt = datetime.fromtimestamp(now_ts)
    session_folder_name = end_dt.strftime("%Y-%m-%d_%H%M%S")
    session_dir = os.path.join(sessions_root, session_folder_name)

    try:
        os.makedirs(session_dir, exist_ok=True)
    except Exception as e:
        print(f"[warning] Failed to create session directory: {e}", file=sys.stderr)
        return

    # Save audio: last N seconds from both sources
    save_mode = (save_audio_mode or "mix").strip().lower()
    if save_mode not in ("separate", "mix", "both"):
        save_mode = "mix"
    mix_gain_mode = (mix_gain_mode or "fixed").strip().lower()
    if mix_gain_mode not in ("fixed", "normalize", "balance"):
        mix_gain_mode = "balance"

    system_audio = np.array([], dtype=np.float32)
    mic_audio = np.array([], dtype=np.float32)
    system_sr = None
    mic_sr = None

    # Collect recent audio
    if system_listener is not None:
        try:
            system_audio = system_listener.get_recent_audio(int(save_audio_seconds))
            system_sr = int(system_listener.samplerate)
        except Exception:
            system_audio = np.array([], dtype=np.float32)
            system_sr = None
    if mic_listener is not None:
        try:
            mic_audio = mic_listener.get_recent_audio(int(save_audio_seconds))
            mic_sr = int(mic_listener.samplerate)
        except Exception:
            mic_audio = np.array([], dtype=np.float32)
            mic_sr = None

    from stt import write_wav_file

    # Separate saves
    system_path = os.path.join(session_dir, "session_system.wav")
    mic_path = os.path.join(session_dir, "session_mic.wav")
    system_saved = False
    mic_saved = False
    if save_mode in ("separate", "both"):
        if system_sr is not None and system_audio is not None and len(system_audio) > 0:
            try:
                write_wav_file(system_path, system_audio, system_sr)
                system_saved = True
            except Exception as e:
                print(f"[warning] Failed to save system audio: {e}", file=sys.stderr)
        else:
            if system_listener is not None:
                print("[warning] No recent system audio to save for the selected interval", file=sys.stderr)
        if mic_sr is not None and mic_audio is not None and len(mic_audio) > 0:
            try:
                write_wav_file(mic_path, mic_audio, mic_sr)
                mic_saved = True
            except Exception as e:
                print(f"[warning] Failed to save microphone audio: {e}", file=sys.stderr)
        else:
            if mic_listener is not None:
                print("[warning] No recent microphone audio to save for the selected interval", file=sys.stderr)

    # Mix mode
    mix_saved = False
    mix_path = os.path.join(session_dir, "session_mix.wav")
    if save_mode in ("mix", "both"):
        try:
            def _resample_linear(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
                if len(x) == 0 or sr_from == sr_to:
                    return x.copy()
                duration = len(x) / float(sr_from)
                n_to = int(round(duration * sr_to))
                if n_to <= 1:
                    return np.zeros((0,), dtype=np.float32)
                # np.interp expects x-axis
                t_from = np.linspace(0.0, duration, num=len(x), endpoint=False)
                t_to = np.linspace(0.0, duration, num=n_to, endpoint=False)
                y = np.interp(t_to, t_from, x.astype(np.float32))
                return y.astype(np.float32)

            def _align_min_by_tail(a: np.ndarray, b: np.ndarray) -> tuple:
                la = len(a)
                lb = len(b)
                if la == 0 or lb == 0:
                    return a, b
                L = min(la, lb)
                return a[-L:], b[-L:]

            have_sys = system_sr is not None and len(system_audio) > 0
            have_mic = mic_sr is not None and len(mic_audio) > 0
            applied_gain = "none"
            if have_sys and have_mic:
                # Prefer system sample rate as target; fallback to mic
                target_sr = int(system_sr if system_sr is not None else mic_sr)
                a = _resample_linear(system_audio, int(system_sr), int(target_sr)) if have_sys else np.zeros((0,), dtype=np.float32)
                b = _resample_linear(mic_audio, int(mic_sr), int(target_sr)) if have_mic else np.zeros((0,), dtype=np.float32)
                a, b = _align_min_by_tail(a, b)
                # Precompute levels
                eps = 1e-9
                peak_a = float(np.max(np.abs(a))) if len(a) > 0 else 0.0
                peak_b = float(np.max(np.abs(b))) if len(b) > 0 else 0.0
                rms_a = float(np.sqrt(np.mean(np.square(a)))) if len(a) > 0 else 0.0
                rms_b = float(np.sqrt(np.mean(np.square(b)))) if len(b) > 0 else 0.0
                if mix_gain_mode == "fixed":
                    g = 0.70710678  # -3 dB per source
                    mix = (a * g) + (b * g)
                    applied_gain = f"fixed(-3dB each; peak_a={peak_a:.3f}, peak_b={peak_b:.3f}, rms_a={rms_a:.3f}, rms_b={rms_b:.3f})"
                else:
                    if mix_gain_mode == "normalize":
                        mix = a + b
                        peak = float(np.max(np.abs(mix))) if len(mix) > 0 else 0.0
                        if peak > 0:
                            mix = (mix / peak * 0.999).astype(np.float32)
                            applied_gain = f"normalize(peak->{0.999}; peak_a={peak_a:.3f}, peak_b={peak_b:.3f}, rms_a={rms_a:.3f}, rms_b={rms_b:.3f})"
                        else:
                            applied_gain = "normalize(no-op)"
                    else:
                        # balance mode: match each RMS to target within bounds, with gate
                        target_lin = float(10.0 ** (float(mix_target_rms_db) / 20.0))
                        max_gain_lin = float(10.0 ** (float(mix_max_gain_db) / 20.0))
                        min_gain_lin = float(10.0 ** (float(mix_min_gain_db) / 20.0))
                        gate_lin = float(10.0 ** (float(mix_rms_gate_db) / 20.0))
                        g_a = target_lin / max(rms_a, eps)
                        g_b = target_lin / max(rms_b, eps)
                        # Do not boost below gate
                        if rms_a < gate_lin and g_a > 1.0:
                            g_a = 1.0
                        if rms_b < gate_lin and g_b > 1.0:
                            g_b = 1.0
                        # Clamp to bounds
                        g_a = float(min(max(g_a, min_gain_lin), max_gain_lin))
                        g_b = float(min(max(g_b, min_gain_lin), max_gain_lin))
                        mix = (a * g_a) + (b * g_b)
                        # Prevent clipping by soft normalization if needed
                        peak_mix = float(np.max(np.abs(mix))) if len(mix) > 0 else 0.0
                        if peak_mix > 1.0:
                            mix = (mix / peak_mix * 0.999).astype(np.float32)
                        applied_gain = f"balance(ga={20*np.log10(g_a):.1f}dB, gb={20*np.log10(g_b):.1f}dB, peak_a={peak_a:.3f}, peak_b={peak_b:.3f}, rms_a={rms_a:.3f}, rms_b={rms_b:.3f})"
                # Final clip to safety
                if len(mix) > 0:
                    mix = np.clip(mix, -1.0, 1.0).astype(np.float32)
                write_wav_file(mix_path, mix, int(target_sr))
                mix_saved = True
            elif have_sys:
                # Only system available; if mode is mix-only, still save as mix for convenience
                target_sr = int(system_sr)
                write_wav_file(mix_path, system_audio, target_sr)
                mix_saved = True
                applied_gain = "passthrough(system)"
            elif have_mic:
                target_sr = int(mic_sr)
                write_wav_file(mix_path, mic_audio, target_sr)
                mix_saved = True
                applied_gain = "passthrough(mic)"
            else:
                print("[warning] No audio from either source to create a mix", file=sys.stderr)
        except Exception as e:
            print(f"[warning] Failed to create/save mixed audio: {e}", file=sys.stderr)

    # Prepare transcript and summary
    transcript_path = os.path.join(session_dir, "transcript.txt")
    # Build source text (limit ~12k chars from tail)
    try:
        text_blocks = []
        if session_text:
            lines = []
            for item in session_text:
                try:
                    if isinstance(item, (tuple, list)) and len(item) >= 3:
                        lines.append(str(item[2]))
                    else:
                        lines.append(str(item))
                except Exception:
                    pass
            text_blocks.append("\n".join([s for s in lines if s]))
        if session_hints:
            # include last up to 10 pairs
            tail = session_hints[-10:] if len(session_hints) > 10 else list(session_hints)
            pairs = []
            for u, a in tail:
                try:
                    pairs.append(f"USER: {u}\nASSISTANT: {a}")
                except Exception:
                    pass
            if pairs:
                text_blocks.append("\n\n".join(pairs))
        source_text_full = "\n\n".join([t for t in text_blocks if t])
        # limit tail ~12000 chars
        max_chars = 12000
        if len(source_text_full) > max_chars:
            source_text = source_text_full[-max_chars:]
        else:
            source_text = source_text_full
    except Exception:
        source_text = ""

    summary_text = ""
    if client is not None and config is not None and source_text.strip():
        try:
            summary_text = make_summary(
                transcript_text=source_text,
                client=client,
                model=config.get("model_hints", "gpt-4o-mini"),
                temperature=float(config.get("temperature", 0.2)),
                system_prompt=config.get("assistant_summary_prompt"),
            )
        except Exception as e:
            print(f"[warning] Summary generation error: {e}", file=sys.stderr)
    try:
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write("System Audio Copilot Session\n")
            f.write("==============================\n\n")
            f.write(f"Start: {start_dt.isoformat()}\n")
            f.write(f"End:   {end_dt.isoformat()}\n")
            try:
                sys_sr_str = (str(system_sr) + " Hz") if system_sr is not None else "n/a"
            except Exception:
                sys_sr_str = "n/a"
            try:
                mic_sr_str = (str(mic_sr) + " Hz") if mic_sr is not None else "n/a"
            except Exception:
                mic_sr_str = "n/a"
            f.write(f"System sample rate: {sys_sr_str}\n")
            f.write(f"Mic sample rate: {mic_sr_str}\n")
            f.write(f"Saved audio seconds: {int(save_audio_seconds)}\n")
            f.write("\n")

            if summary_text:
                f.write("Summary\n")
                f.write("-------\n")
                f.write(summary_text.strip() + "\n\n")

            f.write("Transcript\n")
            f.write("----------\n")
            if session_text:
                for item in session_text:
                    try:
                        if isinstance(item, (tuple, list)) and len(item) >= 3:
                            tstr = str(item[0]).strip()
                            src = str(item[1]).strip().lower()
                            txt = str(item[2]).strip()
                            label = "Me" if src in ("mic", "me") else ("Other" if src in ("system", "other") else "")
                            if label:
                                f.write(f"[{tstr}] [{label}] {txt}\n")
                            else:
                                f.write(f"[{tstr}] {txt}\n")
                        else:
                            f.write(str(item).strip() + "\n")
                    except Exception:
                        pass
            else:
                f.write("(no transcribed text)\n")
            f.write("\n")

            f.write("Assistant\n")
            f.write("---------\n")
            if session_hints:
                for user_text, assistant_text in session_hints:
                    try:
                        f.write("USER: " + user_text.strip() + "\n")
                        f.write("ASSISTANT: " + assistant_text.strip() + "\n")
                        f.write("-" * 20 + "\n")
                    except Exception:
                        pass
            else:
                f.write("(no assistant interactions)\n")
    except Exception as e:
        print(f"[warning] Failed to save transcript: {e}", file=sys.stderr)

    # Print resulting paths
    try:
        print(f"[info] Saved session to: {os.path.abspath(session_dir)}", file=sys.stderr)
        if save_mode in ("separate", "both"):
            if system_saved:
                print(f"[info]  - system audio: {os.path.abspath(system_path)}", file=sys.stderr)
            if mic_saved:
                print(f"[info]  - microphone audio: {os.path.abspath(mic_path)}", file=sys.stderr)
        if save_mode in ("mix", "both") and mix_saved:
            # Print brief summary
            try:
                len_sys = len(system_audio) if system_audio is not None else 0
                len_mic = len(mic_audio) if mic_audio is not None else 0
                print(f"[info]  - mixed audio: {os.path.abspath(mix_path)} (sys_len={len_sys}, sys_sr={system_sr}, mic_len={len_mic}, mic_sr={mic_sr}, gain={applied_gain})", file=sys.stderr)
            except Exception:
                print(f"[info]  - mixed audio: {os.path.abspath(mix_path)}", file=sys.stderr)
        print(f"[info]  - transcript: {os.path.abspath(transcript_path)}", file=sys.stderr)
    except Exception:
        pass

    
    sys.stdout.flush()


def main():
    """Main entry point of the application."""
    parser = argparse.ArgumentParser(
        description="System Audio Copilot - live transcription of system audio with AI hints"
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Transcription interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Sample rate (default: 16000)"
    )
    parser.add_argument(
        "--enter-only",
        action="store_true",
        help="Do not print live transcription, only accumulate for Enter"
    )
    parser.add_argument(
        "--loopback",
        action="store_true",
        default=True,
        help="Capture via WASAPI loopback (listen to the output device, what you hear). Enabled by default"
    )
    parser.add_argument(
        "--no-loopback",
        action="store_true",
        help="Disable system audio (loopback) capture"
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        help="Output device name for loopback (substring, e.g., 'Headphones')"
    )
    parser.add_argument(
        "--input-index",
        type=int,
        default=None,
        help="Explicit input device index (PyAudio) for capture. Useful for selecting [Loopback]"
    )
    # Microphone options
    parser.add_argument(
        "--capture-mic",
        action="store_true",
        default=True,
        help="Enable microphone capture (on by default)"
    )
    parser.add_argument(
        "--no-mic",
        action="store_true",
        help="Disable microphone capture"
    )
    parser.add_argument(
        "--mic-index",
        type=int,
        default=None,
        help="Explicit microphone device index"
    )
    parser.add_argument(
        "--mic-device",
        type=str,
        default=None,
        help="Microphone device name substring"
    )
    parser.add_argument(
        "--mic-samplerate",
        type=int,
        default=None,
        help="Microphone sample rate (default: same as --samplerate)"
    )
    parser.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Microphone channels (1 or 2, default: 1)"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=float(os.getenv("VAD_THRESHOLD", "0.0002")),
        help="Silence threshold (RMS) below which audio is ignored (default: 0.0002)"
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        default=50,
        help="VAD sub-frame size in milliseconds (default: 50)"
    )
    parser.add_argument(
        "--vad-min-voiced-ratio",
        type=float,
        default=0.2,
        help="Minimum fraction of voiced sub-frames to treat the window as speech (default: 0.2)"
    )
    parser.add_argument(
        "--save-on-exit",
        type=str,
        choices=["ask", "yes", "no"],
        default=os.getenv("SAVE_ON_EXIT", "ask"),
        help="Save session on exit: {ask,yes,no} (default: ask)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.getenv("SAVE_DIR", "./sessions"),
        help="Directory to store saved sessions (default: ./sessions)"
    )
    parser.add_argument(
        "--save-audio-seconds",
        type=int,
        default=int(os.getenv("SAVE_AUDIO_SECONDS", "30")),
        help="Number of recent seconds of audio to save (default: 30)"
    )
    parser.add_argument(
        "--buffer-session-seconds",
        type=int,
        default=(int(os.getenv("BUFFER_SESSION_SECONDS")) if os.getenv("BUFFER_SESSION_SECONDS") is not None else None),
        help="Max seconds to keep in RAM for the session ring buffer; 0 = unlimited. If not set, defaults to the value of --save-audio-seconds."
    )
    parser.add_argument(
        "--save-audio-mode",
        type=str,
        choices=["separate", "mix", "both"],
        default=os.getenv("SAVE_AUDIO_MODE", "mix"),
        help="How to save audio on exit: separate files, mixed, or both (default: separate)"
    )
    parser.add_argument(
        "--mix-gain-mode",
        type=str,
        choices=["fixed", "normalize", "balance"],
        default=os.getenv("MIX_GAIN_MODE", "balance"),
        help="Mix gain strategy: fixed (-3 dB), normalize to peak, or balance by RMS"
    )
    parser.add_argument(
        "--mix-target-rms-db",
        type=float,
        default=float(os.getenv("MIX_TARGET_RMS_DB", "-20")),
        help="Target RMS in dBFS for balance mode (default: -20)"
    )
    parser.add_argument(
        "--mix-max-gain-db",
        type=float,
        default=float(os.getenv("MIX_MAX_GAIN_DB", "12")),
        help="Maximum gain in dB per source in balance mode (default: +12)"
    )
    parser.add_argument(
        "--mix-min-gain-db",
        type=float,
        default=float(os.getenv("MIX_MIN_GAIN_DB", "-12")),
        help="Minimum gain in dB per source in balance mode (default: -12)"
    )
    parser.add_argument(
        "--mix-rms-gate-db",
        type=float,
        default=float(os.getenv("MIX_RMS_GATE_DB", "-60")),
        help="Do not boost sources with RMS below this (noise gate) (default: -60)"
    )
    # UI flags
    parser.add_argument(
        "--ui-opacity",
        type=int,
        default=None,
        help="Console window opacity percentage (Windows only, 0-100)"
    )
    parser.add_argument(
        "--no-transparency",
        action="store_true",
        help="Disable console window transparency"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output"
    )
    
    args = parser.parse_args()
    
    # Apply defaults when double-clicking the packaged .exe (no CLI args)
    try:
        is_frozen = bool(getattr(sys, 'frozen', False))
    except Exception:
        is_frozen = False
    if len(sys.argv) <= 1 and is_frozen:
        args.save_on_exit = "yes"
        args.save_audio_seconds = 0
        args.save_audio_mode = "both"
        try:
            args.buffer_session_seconds = 0
        except Exception:
            pass
        if args.ui_opacity is None:
            # If UI_OPACITY provided in .env use it, else fallback to 85
            try:
                env_opacity = int(load_config().get("ui_opacity", 0))
            except Exception:
                env_opacity = 0
            args.ui_opacity = env_opacity if env_opacity > 0 else 85
        args.no_color = False
    
    # Load configuration
    config = load_config()
    client = setup_openai_client(config["api_key"])

    # Prepare ANSI colors (Windows VT enable where possible)
    def _enable_ansi_if_possible() -> bool:
        try:
            if os.name != 'nt':
                return True
            k32 = ctypes.windll.kernel32  # type: ignore
            h = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(h, ctypes.byref(mode)) == 0:
                return False
            new_mode = mode.value | 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if k32.SetConsoleMode(h, new_mode) == 0:
                return False
            return True
        except Exception:
            return False

    # If --ui-opacity not provided, use UI_OPACITY from config if >0
    try:
        if args.ui_opacity is None and int(config.get("ui_opacity", 0)) > 0:
            args.ui_opacity = int(config.get("ui_opacity", 0))
    except Exception:
        pass

    ansi_enabled = (not args.no_color) and _enable_ansi_if_possible()
    config["use_color"] = bool(ansi_enabled)
    if ansi_enabled:
        config["colors"] = {
            "me": "\x1b[95m",        # bright magenta (Me)
            "other": "\x1b[93m",     # bright yellow (Other)
            "assistant": "\x1b[92m", # bright green (Assistant)
            "reset": "\x1b[0m",
        }
    else:
        config["colors"] = {"me": "", "other": "", "assistant": "", "reset": ""}

    # Apply transparency on Windows if requested/allowed
    def _apply_opacity(percent: Optional[int]) -> None:
        try:
            if os.name != 'nt':
                return
            if args.no_transparency:
                return
            if percent is None:
                return
            val = max(0, min(100, int(percent)))
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()  # type: ignore
            if hwnd == 0:
                return
            user32 = ctypes.windll.user32  # type: ignore
            gwl_exstyle = -20
            ws_ex_layered = 0x00080000
            lwa_alpha = 0x2
            # Get existing style
            get_wl = ctypes.windll.user32.GetWindowLongW  # type: ignore
            set_wl = ctypes.windll.user32.SetWindowLongW  # type: ignore
            style = get_wl(hwnd, gwl_exstyle)
            set_wl(hwnd, gwl_exstyle, style | ws_ex_layered)
            alpha = int(round(255 * (val / 100.0)))
            set_attr = ctypes.windll.user32.SetLayeredWindowAttributes  # type: ignore
            set_attr(hwnd, 0, alpha, lwa_alpha)
        except Exception:
            pass
    
    print(f"[info] Starting System Audio Copilot", file=sys.stderr)
    print(f"[info] Transcription interval: {args.window_sec} s", file=sys.stderr)
    print(f"[info] Sample rate: {args.samplerate} Hz", file=sys.stderr)
    print(f"[info] Mode: {'enter-only' if args.enter_only else 'live transcription'}", file=sys.stderr)
    if args.loopback and not args.no_loopback:
        print(f"[info] Audio source: WASAPI loopback", file=sys.stderr)
        if args.output_device:
            print(f"[info] Output device for loopback: {args.output_device}", file=sys.stderr)
    else:
        print(f"[info] Audio source: recording device (microphone/mixer)", file=sys.stderr)
    # Apply requested opacity (Windows; silently ignore elsewhere)
    try:
        _apply_opacity(args.ui_opacity)
    except Exception:
        pass

    print(f"[info] Press Enter for an AI hint, Ctrl+C to exit", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Shutdown coordination flags
    shutdown_requested = threading.Event()
    shutdown_completed = threading.Event()
    shutdown_lock = threading.Lock()

    def graceful_shutdown(reason: str, force_yes: bool = False):
        if shutdown_completed.is_set():
            return
        # Ensure only one thread runs the shutdown body
        with shutdown_lock:
            if shutdown_completed.is_set():
                return
            try:
                shutdown_requested.set()
                # Stop recording cleanly
                try:
                    if audio_listener:
                        audio_listener.stop_recording()
                except Exception:
                    pass
                try:
                    if mic_listener:
                        mic_listener.stop_recording()
                except Exception:
                    pass
                # Save session depending on mode; for non-interactive close auto-yes when asked
                try:
                    effective_mode = str(args.save_on_exit)
                    if force_yes and effective_mode == "ask":
                        effective_mode = "yes"
                    prompt_and_save_session(
                        system_listener=audio_listener,
                        mic_listener=mic_listener,
                        session_text=session_text,
                        session_hints=session_hints,
                        save_on_exit_mode=effective_mode,
                        save_dir_opt=str(args.save_dir),
                        save_audio_seconds=int(args.save_audio_seconds),
                        save_audio_mode=str(args.save_audio_mode),
                        mix_gain_mode=str(args.mix_gain_mode),
                        client=client,
                        config=config,
                        mix_target_rms_db=float(args.mix_target_rms_db),
                        mix_max_gain_db=float(args.mix_max_gain_db),
                        mix_min_gain_db=float(args.mix_min_gain_db),
                        mix_rms_gate_db=float(args.mix_rms_gate_db),
                        session_start_ts=session_start_ts,
                    )
                except Exception as e:
                    print(f"[warning] Failed to save session during shutdown: {e}", file=sys.stderr)
            finally:
                shutdown_completed.set()

    # Initialize audio listener
    enable_system = bool(args.loopback and not args.no_loopback)
    enable_mic = bool(args.capture_mic and not args.no_mic)
    audio_listener = None
    mic_listener = None
    # Determine effective session buffer depth in seconds for in-memory storage
    try:
        buffer_seconds_effective = args.buffer_session_seconds if args.buffer_session_seconds is not None else args.save_audio_seconds
    except Exception:
        buffer_seconds_effective = args.save_audio_seconds

    if enable_system:
        audio_listener = SystemAudioListener(
            samplerate=args.samplerate,
            use_wasapi_loopback=True,
            output_device=args.output_device,
            input_device_index=args.input_index,
            max_session_seconds=int(buffer_seconds_effective)
        )
    if enable_mic:
        mic_listener = MicrophoneListener(
            samplerate=(args.mic_samplerate if args.mic_samplerate else args.samplerate),
            channels=int(args.mic_channels),
            input_device_name=args.mic_device,
            input_device_index=args.mic_index,
            max_session_seconds=int(buffer_seconds_effective)
        )
    
    # Buffer for accumulating text since last Enter
    since_enter_text = []
    since_enter_lock = threading.Lock()
    # Session-wide accumulators
    session_text = []
    session_hints = []
    session_start_ts = time.time()
    
    try:
        # Start audio recording
        if audio_listener:
            audio_listener.start_recording()
        if mic_listener:
            mic_listener.start_recording()
        
        # Start live transcription thread
        threads = []
        if audio_listener:
            t_sys = threading.Thread(
                target=live_transcription_worker,
                args=(
                    audio_listener,
                    client,
                    config,
                    args.window_sec,
                    args.enter_only,
                    args.vad_threshold,
                    args.vad_frame_ms,
                    args.vad_min_voiced_ratio,
                    since_enter_text,
                    since_enter_lock,
                    session_text,
                    "system",
                ),
                daemon=True
            )
            t_sys.start()
            threads.append(t_sys)
        if mic_listener:
            t_mic = threading.Thread(
                target=live_transcription_worker,
                args=(
                    mic_listener,
                    client,
                    config,
                    args.window_sec,
                    args.enter_only,
                    args.vad_threshold,
                    args.vad_frame_ms,
                    args.vad_min_voiced_ratio,
                    since_enter_text,
                    since_enter_lock,
                    session_text,
                    "mic",
                ),
                daemon=True
            )
            t_mic.start()
            threads.append(t_mic)
        # Optional dev watchdog log for stream activity
        if config.get("dev_logs"):
            def _periodic_status():
                while True:
                    try:
                        time.sleep(3.0)
                        if audio_listener:
                            active = audio_listener.is_stream_active()
                            print(f"[debug] system active={active} sr={audio_listener.samplerate} ch={audio_listener.stream_channels}", file=sys.stderr)
                        if mic_listener:
                            active_m = mic_listener.is_stream_active()
                            print(f"[debug] mic    active={active_m} sr={mic_listener.samplerate} ch={mic_listener.stream_channels}", file=sys.stderr)
                    except Exception:
                        break
            threading.Thread(target=_periodic_status, daemon=True).start()

        # Register signal handlers (best-effort cross-platform)
        def _sig_handler(signum, frame):
            if not shutdown_requested.is_set():
                print("\n[info] Signal received, shutting down...", file=sys.stderr)
                graceful_shutdown("signal", force_yes=True)
        try:
            signal.signal(signal.SIGTERM, _sig_handler)
        except Exception:
            pass
        try:
            # SIGHUP not on Windows typically
            signal.signal(getattr(signal, 'SIGHUP', signal.SIGTERM), _sig_handler)
        except Exception:
            pass

        # Windows console close handler via SetConsoleCtrlHandler
        if os.name == 'nt':
            try:
                CTRL_C_EVENT = 0
                CTRL_CLOSE_EVENT = 2
                CTRL_LOGOFF_EVENT = 5
                CTRL_SHUTDOWN_EVENT = 6

                HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)  # type: ignore

                def _console_handler(ctrl_type):
                    try:
                        if ctrl_type in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
                            if not shutdown_requested.is_set():
                                try:
                                    print("\n[info] Console close event, saving session...", file=sys.stderr)
                                except Exception:
                                    pass
                                t = threading.Thread(target=graceful_shutdown, args=("console_close", True), daemon=False)
                                t.start()
                            # Wait briefly for shutdown to complete (Windows gives ~5s)
                            deadline = time.time() + 4.5
                            while not shutdown_completed.is_set() and time.time() < deadline:
                                try:
                                    time.sleep(0.05)
                                except Exception:
                                    break
                            return 1  # handled
                        # Let Ctrl+C be handled normally
                        return 0
                    except Exception:
                        return 1

                _handler = HandlerRoutine(_console_handler)
                # Keep reference to avoid GC
                config["_win_console_handler"] = _handler
                ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, 1)  # type: ignore
            except Exception:
                pass
        
        # Main input loop
        while True:
            try:
                user_input = input()
                if user_input.strip() == "" or user_input.strip() == "Enter":
                    handle_enter_input(since_enter_text, since_enter_lock, client, config, session_hints)
                else:
                    print("[info] Press Enter to get an AI hint")
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\n[info] Termination signal received...", file=sys.stderr)
    except Exception as e:
        print(f"[error] Critical error: {e}", file=sys.stderr)
    finally:
        # Ensure graceful shutdown (idempotent)
        try:
            graceful_shutdown("finally", force_yes=False)
        except Exception as e:
            print(f"[warning] Shutdown error: {e}", file=sys.stderr)
        print("[info] Application finished", file=sys.stderr)


if __name__ == "__main__":
    main()
