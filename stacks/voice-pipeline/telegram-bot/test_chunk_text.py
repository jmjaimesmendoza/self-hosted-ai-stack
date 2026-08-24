"""Checks for chunk_text, shrink_result and STRINGS parity — run directly: python test_chunk_text.py"""
import json
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
from bot import MAX_RESULT_CHARS, MAX_ROWS, STRINGS, chunk_text, shrink_result


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


def test_exact_multiple_has_no_empty_chunk():
    # regression: a line of exactly limit (or k*limit) chars used to emit "" chunks,
    # which Telegram rejects and which dropped the whole answer
    assert chunk_text("x" * 10, 10) == ["x" * 10]
    assert chunk_text("x" * 20, 10) == ["x" * 10, "x" * 10]
    for n in (1, 9, 10, 11, 19, 20, 21):
        assert all(chunk_text("x" * n, 10)), n


def test_strings_locales_share_keys():
    keys = {loc: set(d) for loc, d in STRINGS.items()}
    assert keys["ES"] == keys["EN"] == keys["PT"], keys


def test_shrink_result_small():
    assert json.loads(shrink_result([{"a": 1}])) == [{"a": 1}]


def test_shrink_result_drops_rows_valid_json():
    rows = [{"name": "x" * 100} for _ in range(200)]
    out = shrink_result(rows)
    assert len(out) <= MAX_RESULT_CHARS + 50
    parsed = json.loads(out)  # must stay valid JSON
    assert "of 200 rows" in parsed[-1]


def test_shrink_result_single_giant_row_keeps_data():
    # regression: a single oversized row used to shrink to zero rows (no data at all)
    out = shrink_result([{"blob": "y" * (MAX_RESULT_CHARS * 2)}])
    assert "yyyy" in out and "(row truncated)" in out


def test_shrink_result_notes_row_cap():
    # regression: hitting the MAX_ROWS cursor cap was reported as the true total
    rows = [{"i": i} for i in range(MAX_ROWS)]
    out = shrink_result(rows)
    assert f"at least {MAX_ROWS}" in out or "row cap reached" in out


if __name__ == "__main__":
    test_short_text_single_chunk()
    test_splits_at_line_boundaries()
    test_overlong_line_hard_sliced()
    test_empty_text()
    test_exact_multiple_has_no_empty_chunk()
    test_strings_locales_share_keys()
    test_shrink_result_small()
    test_shrink_result_drops_rows_valid_json()
    test_shrink_result_single_giant_row_keeps_data()
    test_shrink_result_notes_row_cap()
    print("chunk_text + shrink_result + STRINGS: all checks passed")
