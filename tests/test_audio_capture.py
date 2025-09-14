import numpy as np
import pytest
import types

from audio_capture import SystemAudioListener


class DummyStream:
    def __init__(self):
        self.active = True
    def stop_stream(self):
        self.active = False
    def close(self):
        self.active = False


class DummyPyAudio:
    def __init__(self):
        pass
    def get_device_count(self):
        return 1
    def get_device_info_by_index(self, i):
        return {"name": "Dummy", "index": 0, "defaultSampleRate": 16000, "maxInputChannels": 1}
    def get_default_output_device_info(self):
        return {"name": "DummyOut"}
    def get_default_input_device_info(self):
        return {"name": "DummyIn", "index": 0, "defaultSampleRate": 16000, "maxInputChannels": 1}
    def open(self, **kwargs):
        return DummyStream()
    def terminate(self):
        pass


@pytest.fixture(autouse=True)
def patch_pyaudio(monkeypatch):
    import audio_capture as ac
    monkeypatch.setattr(ac, "pyaudio", types.SimpleNamespace(PyAudio=DummyPyAudio, paFloat32=1, paContinue=0, paAbort=1))
    yield


def test_ring_buffer_and_recent_audio(monkeypatch):
    listener = SystemAudioListener(samplerate=16000, channels=1, use_wasapi_loopback=False, max_session_seconds=1)
    # Simulate callback appends by directly manipulating buffers under lock
    with listener.lock:
        for _ in range(20):
            listener.frame_size = 1600
            listener.session_ring_buffer.append(np.ones((1600,), dtype=np.float32))
            listener._update_session_block_limit()
            max_blocks = listener.max_session_blocks
            if len(listener.session_ring_buffer) > max_blocks:
                listener.session_ring_buffer = listener.session_ring_buffer[-max_blocks:]
    recent = listener.get_recent_audio(1)
    assert recent is not None and len(recent) > 0


