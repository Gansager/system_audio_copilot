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

import numpy as np
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from audio_capture import SystemAudioListener
from stt import transcribe_audio_chunk
from llm import make_hint


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
        "dev_logs": str(os.getenv("DEV_LOGS", "0")).strip().lower() in ("1", "true", "yes", "on")
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
    since_enter_lock: threading.Lock
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
                    
                    # Print live transcription if not in enter_only mode
                    if not enter_only:
                        print(f"[live] {transcribed_text}")
                        sys.stdout.flush()
                        
        except Exception as e:
            print(f"[error] Error in transcription thread: {e}", file=sys.stderr)


def handle_enter_input(
    since_enter_text: list,
    since_enter_lock: threading.Lock,
    client: OpenAI,
    config: dict
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
        config["temperature"]
    )
    
    if hint:
        print(f"\n=== ASSISTANT ===")
        print(hint)
        print("=" * 20)
    else:
        print("[error] Failed to get assistant response")
    
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    client = setup_openai_client(config["api_key"])
    
    print(f"[info] Starting System Audio Copilot", file=sys.stderr)
    print(f"[info] Transcription interval: {args.window_sec} s", file=sys.stderr)
    print(f"[info] Sample rate: {args.samplerate} Hz", file=sys.stderr)
    print(f"[info] Mode: {'enter-only' if args.enter_only else 'live transcription'}", file=sys.stderr)
    if args.loopback:
        print(f"[info] Audio source: WASAPI loopback", file=sys.stderr)
        if args.output_device:
            print(f"[info] Output device for loopback: {args.output_device}", file=sys.stderr)
    else:
        print(f"[info] Audio source: recording device (microphone/mixer)", file=sys.stderr)
    print(f"[info] Press Enter for an AI hint, Ctrl+C to exit", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Initialize audio listener
    audio_listener = SystemAudioListener(
        samplerate=args.samplerate,
        use_wasapi_loopback=args.loopback,
        output_device=args.output_device,
        input_device_index=args.input_index
    )
    
    # Buffer for accumulating text since last Enter
    since_enter_text = []
    since_enter_lock = threading.Lock()
    
    try:
        # Start audio recording
        audio_listener.start_recording()
        
        # Start live transcription thread
        transcription_thread = threading.Thread(
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
                since_enter_lock
            ),
            daemon=True
        )
        transcription_thread.start()
        
        # Main input loop
        while True:
            try:
                user_input = input()
                if user_input.strip() == "" or user_input.strip() == "Enter":
                    handle_enter_input(since_enter_text, since_enter_lock, client, config)
                else:
                    print("[info] Press Enter to get an AI hint")
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\n[info] Termination signal received...", file=sys.stderr)
    except Exception as e:
        print(f"[error] Critical error: {e}", file=sys.stderr)
    finally:
        # Stop recording cleanly
        audio_listener.stop_recording()
        print("[info] Application finished", file=sys.stderr)


if __name__ == "__main__":
    main()
