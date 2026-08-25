"""Checks for check_resp — run directly: python test_check_resp.py"""
import os
from unittest.mock import patch

import httpx

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
import bot
from bot import check_resp

REQ = httpx.Request("POST", "http://litellm:4000/v1/chat/completions")


def test_error_logs_body_and_raises():
    resp = httpx.Response(400, text="upstream model 'nope' not found", request=REQ)
    with patch.object(bot, "logger") as log:
        try:
            check_resp(resp, "litellm/tool")
            assert False, "expected HTTPStatusError"
        except httpx.HTTPStatusError:
            pass
    logged = log.error.call_args[0][0]
    assert "upstream model 'nope' not found" in logged  # the body httpx would have dropped
    assert "litellm/tool" in logged and "400" in logged
    assert not log.debug.called


def test_ok_logs_body_at_debug_and_returns():
    resp = httpx.Response(200, text='{"choices": []}', request=REQ)
    with patch.object(bot, "logger") as log:
        check_resp(resp, "whisper")  # must not raise
    assert '{"choices": []}' in log.debug.call_args[0][0]
    assert not log.error.called


def test_long_body_truncated():
    resp = httpx.Response(500, text="x" * (bot.LOG_BODY_CHARS + 500), request=REQ)
    with patch.object(bot, "logger") as log:
        try:
            check_resp(resp, "litellm/final")
        except httpx.HTTPStatusError:
            pass
    assert log.error.call_args[0][0].count("x") == bot.LOG_BODY_CHARS


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
