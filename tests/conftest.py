import os
import sys

# Ensure project root is on sys.path so tests can import modules like 'audio_capture'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


