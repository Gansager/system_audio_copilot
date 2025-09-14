import numpy as np
import io
import soundfile as sf
import pytest

from stt import pcm_float_to_wav_bytes, write_wav_file, transcribe_audio_chunk


def test_pcm_float_to_wav_bytes_and_write(tmp_path):
    sr = 16000
    t = np.linspace(0, 0.01, int(sr*0.01), endpoint=False)
    x = (0.5*np.sin(2*np.pi*440*t)).astype(np.float32)
    wav = pcm_float_to_wav_bytes(x, sr)
    assert isinstance(wav, (bytes, bytearray)) and len(wav) > 0
    out = tmp_path/"a.wav"
    write_wav_file(str(out), x, sr)
    data, r = sf.read(str(out))
    assert r == sr
    assert data.dtype.kind in ("i", "f")


class DummyClient:
    class audio:
        class transcriptions:
            @staticmethod
            def create(model, file, response_format, language=None):
                # Echo back a constant text regardless of input
                return "hello"


def test_transcribe_audio_chunk_language_hint(monkeypatch):
    sr = 16000
    x = np.zeros((sr//10,), dtype=np.float32)
    out = transcribe_audio_chunk(x, sr, DummyClient(), model="whisper-1", language="de")
    assert out == "hello"


