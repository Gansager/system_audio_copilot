#!/usr/bin/env python3
"""
System Audio Copilot - CLI инструмент для живой транскрибации системного звука
с возможностью получения подсказок от AI ассистента.
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from audio_capture import SystemAudioListener
from stt import transcribe_audio_chunk
from llm import make_hint


def load_config():
    """Загрузка конфигурации из переменных окружения."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY не найден в переменных окружения", file=sys.stderr)
        sys.exit(1)
    
    return {
        "api_key": api_key,
        "model_transcribe": os.getenv("MODEL_TRANSCRIBE", "whisper-1"),
        "model_hints": os.getenv("MODEL_HINTS", "gpt-4o-mini"),
        "temperature": float(os.getenv("TEMPERATURE", "0.2"))
    }


def setup_openai_client(api_key: str) -> OpenAI:
    """Создание OpenAI клиента."""
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[error] Не удалось создать OpenAI клиент: {e}", file=sys.stderr)
        sys.exit(1)


def live_transcription_worker(
    audio_listener: SystemAudioListener,
    client: OpenAI,
    config: dict,
    window_sec: float,
    enter_only: bool,
    since_enter_text: list,
    since_enter_lock: threading.Lock
):
    """
    Фоновый поток для живой транскрибации.
    
    Args:
        audio_listener: Слушатель аудио
        client: OpenAI клиент
        config: Конфигурация
        window_sec: Интервал транскрибации в секундах
        enter_only: Флаг режима "только по Enter"
        since_enter_text: Список для накопления текста
        since_enter_lock: Блокировка для since_enter_text
    """
    while True:
        try:
            time.sleep(window_sec)
            
            # Получаем аудио чанк
            audio_chunk = audio_listener.get_chunk_and_clear()
            
            if len(audio_chunk) > 0:
                # Транскрибируем
                transcribed_text = transcribe_audio_chunk(
                    audio_chunk, 
                    audio_listener.samplerate, 
                    client, 
                    config["model_transcribe"]
                )
                
                if transcribed_text:
                    # Добавляем в буфер для Enter
                    with since_enter_lock:
                        since_enter_text.append(transcribed_text)
                    
                    # Печатаем live транскрипцию если не в режиме enter_only
                    if not enter_only:
                        print(f"[live] {transcribed_text}")
                        sys.stdout.flush()
                        
        except Exception as e:
            print(f"[error] Ошибка в потоке транскрибации: {e}", file=sys.stderr)


def handle_enter_input(
    since_enter_text: list,
    since_enter_lock: threading.Lock,
    client: OpenAI,
    config: dict
):
    """
    Обработка ввода Enter для получения подсказки от AI.
    
    Args:
        since_enter_text: Список накопленного текста
        since_enter_lock: Блокировка для since_enter_text
        client: OpenAI клиент
        config: Конфигурация
    """
    with since_enter_lock:
        if not since_enter_text:
            print("[warning] Нет накопленного текста для отправки")
            return
        
        # Объединяем весь накопленный текст
        full_text = " ".join(since_enter_text).strip()
        
        # Очищаем буфер
        since_enter_text.clear()
    
    if not full_text:
        print("[warning] Нет накопленного текста для отправки")
        return
    
    print(f"[sending] Отправляем текст: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
    
    # Получаем подсказку от AI
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
        print("[error] Не удалось получить ответ от ассистента")
    
    sys.stdout.flush()


def main():
    """Основная функция приложения."""
    parser = argparse.ArgumentParser(
        description="System Audio Copilot - живая транскрибация системного звука с AI подсказками"
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Интервал транскрибации в секундах (по умолчанию: 3.0)"
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Частота дискретизации (по умолчанию: 16000)"
    )
    parser.add_argument(
        "--enter-only",
        action="store_true",
        help="Не печатать live транскрипцию, только накапливать для Enter"
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config()
    client = setup_openai_client(config["api_key"])
    
    print(f"[info] Запуск System Audio Copilot", file=sys.stderr)
    print(f"[info] Интервал транскрибации: {args.window_sec} сек", file=sys.stderr)
    print(f"[info] Частота дискретизации: {args.samplerate} Гц", file=sys.stderr)
    print(f"[info] Режим: {'только по Enter' if args.enter_only else 'живая транскрипция'}", file=sys.stderr)
    print(f"[info] Нажмите Enter для получения подсказки от AI, Ctrl+C для выхода", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Инициализируем слушатель аудио
    audio_listener = SystemAudioListener(samplerate=args.samplerate)
    
    # Буфер для накопления текста с последнего Enter
    since_enter_text = []
    since_enter_lock = threading.Lock()
    
    try:
        # Запускаем запись аудио
        audio_listener.start_recording()
        
        # Запускаем поток живой транскрибации
        transcription_thread = threading.Thread(
            target=live_transcription_worker,
            args=(audio_listener, client, config, args.window_sec, args.enter_only, since_enter_text, since_enter_lock),
            daemon=True
        )
        transcription_thread.start()
        
        # Основной цикл обработки ввода
        while True:
            try:
                user_input = input()
                if user_input.strip() == "" or user_input.strip() == "Enter":
                    handle_enter_input(since_enter_text, since_enter_lock, client, config)
                else:
                    print("[info] Нажмите Enter для получения подсказки от AI")
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\n[info] Получен сигнал завершения...", file=sys.stderr)
    except Exception as e:
        print(f"[error] Критическая ошибка: {e}", file=sys.stderr)
    finally:
        # Корректно останавливаем запись
        audio_listener.stop_recording()
        print("[info] Приложение завершено", file=sys.stderr)


if __name__ == "__main__":
    main()
