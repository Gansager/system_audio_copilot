"""
Module for transcribing audio via OpenAI Whisper API.
"""

import io
import sys
import numpy as np
import soundfile as sf
from openai import OpenAI
from typing import Optional


def pcm_float_to_wav_bytes(audio_f32: np.ndarray, sr: int) -> bytes:
    """
    Convert float32 audio data to WAV bytes.
    
    Args:
        audio_f32: Audio data in float32 format
        sr: Sample rate
    
    Returns:
        bytes: WAV file as bytes
    """
    if len(audio_f32) == 0:
        return b""
    
    # Convert float32 to int16
    # Clamp values to [-1.0, 1.0]
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    
    # Convert to int16
    audio_int16 = (audio_f32 * 32767).astype(np.int16)
    
    # Create an in-memory WAV file
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_int16, sr, format='WAV', subtype='PCM_16')
    wav_bytes = wav_buffer.getvalue()
    wav_buffer.close()
    
    return wav_bytes


def transcribe_wav_bytes(wav_bytes: bytes, client: OpenAI, model: str = "whisper-1") -> str:
    """
    Transcribe WAV bytes via the OpenAI Whisper API.
    
    Args:
        wav_bytes: WAV file as bytes
        client: OpenAI client
        model: Whisper model to use
    
    Returns:
        str: Recognized text
    """
    if len(wav_bytes) == 0:
        return ""
    
    try:
        # Create a temp file in memory
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"
        
        # Send for transcription
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text"
        )
        
        # Return stripped text
        return response.strip() if response else ""
        
    except Exception as e:
        print(f"[error] Transcription error: {e}", file=sys.stderr)
        return ""


def transcribe_audio_chunk(audio_f32: np.ndarray, sr: int, client: OpenAI, model: str = "whisper-1") -> str:
    """
    Transcribe an audio chunk (full pipeline).
    
    Args:
        audio_f32: Audio data in float32 format
        sr: Sample rate
        client: OpenAI client
        model: Whisper model to use
    
    Returns:
        str: Recognized text
    """
    if len(audio_f32) == 0:
        return ""
    
    # Convert to WAV bytes
    wav_bytes = pcm_float_to_wav_bytes(audio_f32, sr)
    
    if len(wav_bytes) == 0:
        return ""
    
    # Transcribe
    return transcribe_wav_bytes(wav_bytes, client, model)
