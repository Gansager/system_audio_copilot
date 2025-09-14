import builtins
import types

import pytest

from main import filter_allowed_text, text_is_banned


def test_filter_allowed_text_cyrillic_kept():
    txt = "Привет, как дела?"
    out = filter_allowed_text(txt, ["ru", "en", "de", "uk"])
    assert out == txt


def test_filter_allowed_text_german_diacritics_kept():
    txt = "Schönen Grüße für äußere Häuser"
    out = filter_allowed_text(txt, ["de"])  # allow de only
    assert "ö" in out and "ä" in out and "ü" in out


def test_filter_allowed_text_other_diacritics_removed():
    txt = "Español: acción rápido señor"
    out = filter_allowed_text(txt, ["en", "de"])  # not spanish diacritics
    # 80% letters rule likely drops to empty
    assert out == ""


def test_filter_allowed_text_mixed_below_threshold_dropped():
    txt = "你好 hello привет"
    out = filter_allowed_text(txt, ["en", "ru"])  # mixed with CJK
    assert out == ""


def test_text_is_banned_substring_case_insensitive():
    cfg = {"_banned_norm": ["thankyousomuchforwatching", "субтитрыдобавилdimatorzok"]}
    assert text_is_banned("Thank you so much for watching!", cfg)
    assert text_is_banned("Субтитры добавил DimaTorzok", cfg)
    assert not text_is_banned("legit phrase", cfg)


