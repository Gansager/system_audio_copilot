import pytest

from llm import make_hint, make_summary, DEFAULT_SYSTEM_PROMPT, DEFAULT_SUMMARY_SYSTEM_PROMPT


class DummyChat:
    class choices_item:
        def __init__(self, content):
            self.message = type("m", (), {"content": content})

    def __init__(self, text):
        self.choices = [self.choices_item(text)]


class DummyClient:
    class chat:
        class completions:
            @staticmethod
            def create(model, temperature, messages):
                # Ensure system prompt contains language restriction
                sys_msg = messages[0]["content"]
                assert any(lang in sys_msg.lower() for lang in ["russian", "english", "german", "ukrainian"]) or "language of the meeting" in sys_msg.lower()
                return DummyChat("ok")


def test_make_hint_and_summary(monkeypatch):
    c = DummyClient()
    assert make_hint("hello", c, model="gpt-x") == "ok"
    assert make_summary("hello", c, model="gpt-x", temperature=0.0) == "ok"


