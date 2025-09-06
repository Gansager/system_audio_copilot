"""
Модуль для транскрибации аудио через OpenAI Whisper API.
"""

import io
import sys
import numpy as np
import soundfile as sf
from openai import OpenAI
from typing import Optional


def pcm_float_to_wav_bytes(audio_f32: np.ndarray, sr: int) -> bytes:
    """
    Конвертировать float32 аудио данные в WAV байты.
    
    Args:
        audio_f32: Аудио данные в формате float32
        sr: Частота дискретизации
    
    Returns:
        bytes: WAV файл в виде байтов
    """
    if len(audio_f32) == 0:
        return b""
    
    # Конвертируем float32 в int16
    # Ограничиваем значения в диапазоне [-1.0, 1.0]
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    
    # Конвертируем в int16
    audio_int16 = (audio_f32 * 32767).astype(np.int16)
    
    # Создаем WAV файл в памяти
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_int16, sr, format='WAV', subtype='PCM_16')
    wav_bytes = wav_buffer.getvalue()
    wav_buffer.close()
    
    return wav_bytes


def transcribe_wav_bytes(wav_bytes: bytes, client: OpenAI, model: str = "whisper-1") -> str:
    """
    Транскрибировать WAV байты через OpenAI Whisper API.
    
    Args:
        wav_bytes: WAV файл в виде байтов
        client: OpenAI клиент
        model: Модель Whisper для использования
    
    Returns:
        str: Распознанный текст
    """
    if len(wav_bytes) == 0:
        return ""
    
    try:
        # Создаем временный файл в памяти
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"
        
        # Отправляем на транскрибацию
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text"
        )
        
        # Возвращаем очищенный текст
        return response.strip() if response else ""
        
    except Exception as e:
        print(f"[error] Ошибка транскрибации: {e}", file=sys.stderr)
        return ""


def transcribe_audio_chunk(audio_f32: np.ndarray, sr: int, client: OpenAI, model: str = "whisper-1") -> str:
    """
    Транскрибировать аудио чанк (полный pipeline).
    
    Args:
        audio_f32: Аудио данные в формате float32
        sr: Частота дискретизации
        client: OpenAI клиент
        model: Модель Whisper для использования
    
    Returns:
        str: Распознанный текст
    """
    if len(audio_f32) == 0:
        return ""
    
    # Конвертируем в WAV байты
    wav_bytes = pcm_float_to_wav_bytes(audio_f32, sr)
    
    if len(wav_bytes) == 0:
        return ""
    
    # Транскрибируем
    return transcribe_wav_bytes(wav_bytes, client, model)
