import sys
import os

def main():
    try:
        import pyaudio  # type: ignore
    except Exception as e:
        if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
            print(f"[warning] PyAudio not available: {e}")
        pyaudio = None  # type: ignore

    try:
        import sounddevice as sd  # type: ignore
    except Exception as e:
        if os.getenv("DEV_LOGS", "0").strip().lower() in ("1", "true", "yes", "on"):
            print(f"[warning] sounddevice not available: {e}")
        sd = None  # type: ignore

    if pyaudio is not None:
        try:
            p = pyaudio.PyAudio()
            print("=== PyAudio devices ===")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                try:
                    host_name = p.get_host_api_info_by_index(info['hostApi'])['name']
                except Exception:
                    host_name = str(info.get('hostApi'))
                name = str(info.get('name', ''))
                in_ch = int(info.get('maxInputChannels', 0))
                out_ch = int(info.get('maxOutputChannels', 0))
                try:
                    sr = int(info.get('defaultSampleRate', 0))
                except Exception:
                    sr = 0
                print(f"{i}: {name} | hostApi={host_name} | in={in_ch} out={out_ch} | defaultSR={sr}")
            try:
                print('Default in :', p.get_default_input_device_info().get('name', '-'))
            except Exception:
                print('Default in : -')
            try:
                print('Default out:', p.get_default_output_device_info().get('name', '-'))
            except Exception:
                print('Default out: -')
            p.terminate()
        except Exception as e:
            print(f"[error] Failed to enumerate PyAudio devices: {e}")

    if sd is not None:
        try:
            print("\n=== sounddevice devices ===")
            for i, d in enumerate(sd.query_devices()):
                name = d.get('name', '')
                in_ch = d.get('max_input_channels', 0)
                out_ch = d.get('max_output_channels', 0)
                print(f"{i}: {name} | in={in_ch} out={out_ch}")
            try:
                print('Default devices:', sd.default.device)
            except Exception:
                pass
        except Exception as e:
            print(f"[error] Failed to enumerate sounddevice devices: {e}")


if __name__ == "__main__":
    # Hint for Windows console encoding
    if sys.platform == 'win32':
        print("Tip: In CMD run 'chcp 65001' first for proper Unicode output.")
    main()


