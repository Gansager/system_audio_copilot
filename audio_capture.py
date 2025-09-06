"""
Модуль для захвата системного звука через WASAPI loopback.
"""

import sys
import threading
import time
import numpy as np
import pyaudio
from typing import Optional


class SystemAudioListener:
    """
    Класс для захвата системного звука через WASAPI loopback.
    Использует pyaudio для захвата системного звука.
    """
    
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        """
        Инициализация слушателя системного звука.
        
        Args:
            samplerate: Частота дискретизации (по умолчанию 16 кГц)
            channels: Количество каналов (по умолчанию 1 - моно)
        """
        self.samplerate = samplerate
        self.channels = channels
        self.frame_duration_ms = 100  # Длительность кадра в миллисекундах
        self.frame_size = int(samplerate * self.frame_duration_ms / 1000)
        
        # Буферы для аудио данных
        self.live_buffer = []  # Для живой транскрибации
        self.since_enter_buffer = []  # Для отправки по Enter
        
        # Потокобезопасность
        self.lock = threading.Lock()
        
        # Состояние
        self.is_recording = False
        self.audio = None
        self.stream = None
        self.recording_thread = None
        
        # Инициализация pyaudio
        try:
            self.audio = pyaudio.PyAudio()
            print(f"[info] PyAudio инициализирован", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Не удалось инициализировать PyAudio: {e}")
    
    def start_recording(self):
        """Запуск записи системного звука."""
        if self.is_recording:
            return
        
        try:
            # Находим устройство для loopback (обычно это стерео микшер)
            loopback_device = None
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                device_name = device_info['name'].lower()
                # Ищем устройство loopback или стерео микшер
                if ('stereo mix' in device_name or 
                    'loopback' in device_name or 
                    'what u hear' in device_name or
                    'wave out mix' in device_name):
                    loopback_device = i
                    print(f"[info] Найдено устройство loopback: {device_info['name']}", file=sys.stderr)
                    break
            
            if loopback_device is None:
                # Если не нашли специальное устройство, используем устройство по умолчанию
                loopback_device = self.audio.get_default_input_device_info()['index']
                print(f"[info] Используется устройство по умолчанию: {self.audio.get_device_info_by_index(loopback_device)['name']}", file=sys.stderr)
            
            # Создаем поток для записи
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.samplerate,
                input=True,
                input_device_index=loopback_device,
                frames_per_buffer=self.frame_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            print("[info] Запись системного звука запущена", file=sys.stderr)
            
        except Exception as e:
            raise RuntimeError(f"Не удалось запустить запись: {e}")
    
    def stop_recording(self):
        """Остановка записи системного звука."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        print("[info] Запись системного звука остановлена", file=sys.stderr)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback функция для обработки аудио данных."""
        try:
            if status:
                print(f"[warning] Статус аудио потока: {status}", file=sys.stderr)
            
            if in_data:
                # Конвертируем байты в numpy массив float32
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Если стерео, конвертируем в моно
                if self.channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                with self.lock:
                    # Добавляем в оба буфера
                    self.live_buffer.append(audio_data.copy())
                    self.since_enter_buffer.append(audio_data.copy())
                    
                    # Ограничиваем размер буферов (храним последние 30 секунд)
                    max_frames = int(30 * self.samplerate / self.frame_size)
                    if len(self.live_buffer) > max_frames:
                        self.live_buffer = self.live_buffer[-max_frames:]
                    if len(self.since_enter_buffer) > max_frames:
                        self.since_enter_buffer = self.since_enter_buffer[-max_frames:]
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            print(f"[error] Ошибка в audio callback: {e}", file=sys.stderr)
            return (None, pyaudio.paAbort)
    
    def get_chunk_and_clear(self) -> np.ndarray:
        """
        Получить накопленные данные для живой транскрибации и очистить буфер.
        
        Returns:
            numpy.ndarray: Аудио данные в формате float32
        """
        with self.lock:
            if not self.live_buffer:
                return np.array([], dtype=np.float32)
            
            # Объединяем все кадры в один массив
            audio_chunk = np.concatenate(self.live_buffer, axis=0)
            
            # Очищаем буфер
            self.live_buffer.clear()
            
            return audio_chunk
    
    def get_since_last_enter_and_clear(self) -> np.ndarray:
        """
        Получить накопленные данные с последнего Enter и очистить буфер.
        
        Returns:
            numpy.ndarray: Аудио данные в формате float32
        """
        with self.lock:
            if not self.since_enter_buffer:
                return np.array([], dtype=np.float32)
            
            # Объединяем все кадры в один массив
            audio_chunk = np.concatenate(self.since_enter_buffer, axis=0)
            
            # Очищаем буфер
            self.since_enter_buffer.clear()
            
            return audio_chunk
    
    def __enter__(self):
        """Контекстный менеджер - вход."""
        self.start_recording()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - выход."""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()


