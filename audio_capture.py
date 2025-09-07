"""
Модуль для захвата системного звука через WASAPI loopback.
"""

import sys
import threading
import time
import numpy as np
import pyaudio
from typing import Optional

# Опциональный импорт sounddevice для WASAPI loopback
try:
    import sounddevice as sd  # type: ignore
    HAVE_SD = True
except Exception:
    sd = None  # type: ignore
    HAVE_SD = False

# Опциональный импорт PyAudioWPatch (WASAPI loopback)
try:
    import pyaudiowpatch as pyaudio_wasapi  # type: ignore
    HAVE_PAW = True
except Exception:
    pyaudio_wasapi = None  # type: ignore
    HAVE_PAW = False


class SystemAudioListener:
    """
    Класс для захвата системного звука через WASAPI loopback.
    Использует pyaudio для захвата системного звука.
    """
    
    def __init__(self, samplerate: int = 16000, channels: int = 1, use_wasapi_loopback: bool = False, output_device: Optional[str] = None, input_device_index: Optional[int] = None):
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
        self.stream = None  # Поток PyAudio/PyAudioWPatch или sounddevice
        self.backend = "pyaudio"  # "pyaudio" | "sounddevice"
        self.pa_module = pyaudio  # активный модуль PyAudio (стандартный или patched)
        self._pa_continue = pyaudio.paContinue
        self._pa_abort = pyaudio.paAbort
        self.stream_channels = channels  # фактическое число каналов в открытом потоке

        # Настройки loopback
        self.use_wasapi_loopback = bool(use_wasapi_loopback)
        self.output_device_name = output_device
        self.input_device_index = input_device_index
        self.recording_thread = None
        
        # Инициализация pyaudio (нужен даже если будем пробовать sounddevice, чтобы сохранить обратную совместимость)
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
            # Предпочитаем WASAPI loopback на Windows
            if self.use_wasapi_loopback and sys.platform == 'win32':
                # 1) PyAudioWPatch (надежный вариант)
                if HAVE_PAW:
                    self._start_recording_pyaudio_wasapi_loopback()
                    return
                # 2) sounddevice (если доступен)
                if HAVE_SD:
                    self._start_recording_sounddevice_loopback()
                    return

            # Иначе используем прежний путь через PyAudio (микрофон или loopback-устройство в списке входов)
            loopback_device = None
            sel_device_info = None

            desired_out_name_lower = None
            if self.use_wasapi_loopback:
                try:
                    if self.output_device_name:
                        desired_out_name_lower = self.output_device_name.lower()
                    else:
                        out_info = self.audio.get_default_output_device_info()
                        desired_out_name_lower = str(out_info.get('name', '')).lower()
                except Exception:
                    desired_out_name_lower = None

            first_loopback_idx = None
            first_loopback_info = None

            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                device_name = str(device_info.get('name', '')).lower()
                # Ищем входные устройства с пометкой loopback
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
                        print(f"[info] Найдено устройство loopback: {device_info['name']}", file=sys.stderr)
                        break

            if loopback_device is None and first_loopback_idx is not None:
                loopback_device = first_loopback_idx
                sel_device_info = first_loopback_info
                print(f"[info] Найдено устройство loopback: {sel_device_info['name']}", file=sys.stderr)

            if loopback_device is None:
                sel_device_info = self.audio.get_default_input_device_info()
                loopback_device = sel_device_info['index']
                print(f"[info] Используется устройство по умолчанию: {self.audio.get_device_info_by_index(loopback_device)['name']}", file=sys.stderr)

            # Если пользователь явно указал индекс входного устройства — используем его
            if self.input_device_index is not None:
                try:
                    sel_device_info = self.audio.get_device_info_by_index(self.input_device_index)
                    loopback_device = self.input_device_index
                    print(f"[info] Используется указанный индекс устройства: {sel_device_info['name']} (#{self.input_device_index})", file=sys.stderr)
                except Exception as e:
                    print(f"[warning] Указанный индекс входного устройства недоступен: {e}. Игнорируем.", file=sys.stderr)

            # Подгоняем частоту дискретизации и каналы под устройство
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
            print("[info] Запись системного звука запущена", file=sys.stderr)

        except Exception as e:
            raise RuntimeError(f"Не удалось запустить запись: {e}")

    def _start_recording_sounddevice_loopback(self):
        """Запуск записи через sounddevice в режиме WASAPI loopback."""
        assert HAVE_SD
        try:
            # Определяем устройство вывода
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
                    # Ищем первое доступное устройство вывода
                    for idx, info in enumerate(devices):
                        if info.get('max_output_channels', 0) > 0:
                            selected_output_index = idx
                            selected_output_name = info.get('name', str(idx))
                            break
                if selected_output_index is None:
                    raise RuntimeError("Не найдено доступное устройство вывода для loopback")

            # Создаем входной поток c loopback=True на устройстве вывода
            wasapi_settings = sd.WasapiSettings(loopback=True)  # type: ignore

            # Для стабильности используем 2 канала и затем миксуем в моно при необходимости
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
            print(f"[info] WASAPI loopback активирован. Устройство вывода: {selected_output_name}", file=sys.stderr)

        except Exception as e:
            print(f"[warning] Не удалось запустить WASAPI loopback: {e}. Переход на PyAudio.", file=sys.stderr)
            # Фолбэк на PyAudio
            self.use_wasapi_loopback = False
            self._safe_close_sd_stream()
            # Повторный запуск через PyAudio
            self.start_recording()

    def _start_recording_pyaudio_wasapi_loopback(self):
        """Запуск записи через PyAudioWPatch (WASAPI loopback)."""
        assert HAVE_PAW
        try:
            p = pyaudio_wasapi.PyAudio()  # type: ignore
            # Сохраняем как активный PyAudio
            if self.audio is not None:
                try:
                    self.audio.terminate()
                except Exception:
                    pass
            self.audio = p

            # Ищем host API WASAPI
            wasapi_info = p.get_host_api_info_by_type(pyaudio_wasapi.paWASAPI)  # type: ignore
            wasapi_index = wasapi_info.get('index', None)
            if wasapi_index is None:
                raise RuntimeError("WASAPI host API недоступен")

            selected_output_index = None
            selected_output_name = None

            # Если задано имя устройства, ищем среди WASAPI output устройств
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

            # Иначе берём устройство вывода по умолчанию для WASAPI
            if selected_output_index is None:
                default_out = wasapi_info.get('defaultOutputDevice', -1)
                if isinstance(default_out, int) and default_out >= 0:
                    dev = p.get_device_info_by_index(default_out)
                    selected_output_index = dev['index']
                    selected_output_name = dev.get('name', str(default_out))
                else:
                    # Фолбэк: первое доступное output устройство в WASAPI
                    for i in range(p.get_device_count()):
                        dev = p.get_device_info_by_index(i)
                        if dev.get('hostApi') == wasapi_index and dev.get('maxOutputChannels', 0) > 0:
                            selected_output_index = dev['index']
                            selected_output_name = dev.get('name', str(i))
                            break

            if selected_output_index is None:
                raise RuntimeError("Не найдено WASAPI устройство вывода для loopback")

            # Частота дискретизации по умолчанию для выбранного устройства
            dev_info = p.get_device_info_by_index(selected_output_index)
            try:
                dev_sr = int(dev_info.get('defaultSampleRate', self.samplerate))  # type: ignore
            except Exception:
                dev_sr = self.samplerate
            self.samplerate = dev_sr

            # Для надежности откроем 2 канала, затем при необходимости сведём в моно
            self.stream_channels = 2 if self.channels == 1 else max(1, self.channels)

            # Открываем входной поток как loopback с выбранного устройства вывода (через StreamInfo)
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
            print(f"[info] WASAPI loopback активирован (PyAudioWPatch). Устройство вывода: {selected_output_name}", file=sys.stderr)
            print(f"[info] Эффективная частота: {self.samplerate} Гц, каналы: {self.stream_channels}", file=sys.stderr)

        except Exception as e:
            print(f"[warning] Не удалось запустить WASAPI loopback (PyAudioWPatch): {e}. Переход на резервные режимы.", file=sys.stderr)
            # Попробуем sounddevice
            if HAVE_SD:
                try:
                    self._start_recording_sounddevice_loopback()
                    return
                except Exception:
                    pass
            # Иначе фолбэк на PyAudio обычный
            self.use_wasapi_loopback = False
            self.start_recording()
    
    def stop_recording(self):
        """Остановка записи системного звука."""
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
            
            return (None, self._pa_continue)
            
        except Exception as e:
            print(f"[error] Ошибка в audio callback: {e}", file=sys.stderr)
            return (None, self._pa_abort)

    def _sd_callback(self, indata, frames, time_info, status):  # type: ignore
        """Callback для sounddevice (WASAPI loopback)."""
        try:
            if status:
                print(f"[warning] Статус аудио потока (sd): {status}", file=sys.stderr)

            if indata is not None:
                audio_data = np.array(indata, dtype=np.float32, copy=False)
                # Приводим к (N,) float32
                if audio_data.ndim == 2:
                    if self.channels == 1:
                        audio_data = audio_data.mean(axis=1)
                    else:
                        # Оставляем как есть, затем выровняем в 1D
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
            print(f"[error] Ошибка в sd callback: {e}", file=sys.stderr)

    def _safe_close_sd_stream(self):
        try:
            if self.stream is not None and self.backend == "sounddevice":
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
    
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


