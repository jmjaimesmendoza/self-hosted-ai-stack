"""Checks for fmt_timings — run directly: python test_fmt_timings.py"""
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
from bot import fmt_timings


def test_multi_entry_shows_sum():
    out = fmt_timings({"llm": [1.234, 0.981]})
    assert "llm: 1.23s + 0.98s = 2.21s" in out, out
    assert out.endswith("total: 2.21s"), out


def test_single_entry_no_sum():
    out = fmt_timings({"db": [0.5]})
    assert "db: 0.50s" in out, out
    assert "=" not in out, out


def test_empty_interval_skipped():
    out = fmt_timings({"llm": [1.0], "db": []})
    assert "db" not in out, out


def test_total_spans_intervals():
    out = fmt_timings({"llm": [1.0, 1.0], "db": [0.5], "format": [0.25]})
    assert out.endswith("total: 2.75s"), out


if __name__ == "__main__":
    test_multi_entry_shows_sum()
    test_single_entry_no_sum()
    test_empty_interval_skipped()
    test_total_spans_intervals()
    print("fmt_timings: all checks passed")
