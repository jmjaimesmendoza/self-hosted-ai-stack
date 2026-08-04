"""Checks for chunk_text and STRINGS parity — run directly: python test_chunk_text.py"""
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
from bot import STRINGS, chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello\nworld", 100) == ["hello\nworld"]


def test_splits_at_line_boundaries():
    chunks = chunk_text("aaa\nbbb\nccc", 7)
    assert chunks == ["aaa\nbbb", "ccc"], chunks
    assert all(len(c) <= 7 for c in chunks)


def test_overlong_line_hard_sliced():
    chunks = chunk_text("x" * 25, 10)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5], chunks


def test_empty_text():
    assert chunk_text("", 10) == [""]


def test_strings_locales_share_keys():
    keys = {loc: set(d) for loc, d in STRINGS.items()}
    assert keys["ES"] == keys["EN"] == keys["PT"], keys


if __name__ == "__main__":
    test_short_text_single_chunk()
    test_splits_at_line_boundaries()
    test_overlong_line_hard_sliced()
    test_empty_text()
    test_strings_locales_share_keys()
    print("chunk_text + STRINGS: all checks passed")
