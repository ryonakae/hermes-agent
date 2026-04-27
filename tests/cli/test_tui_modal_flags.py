"""TUI modal detection and simple approval opt-in (overlay vs stdin)."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


def _bare_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._sudo_state = None
    cli._clarify_state = None
    cli._secret_state = None
    cli._model_picker_state = None
    cli._clarify_freetext = False
    return cli


class TestTuiModalActive:
    def test_false_when_idle(self):
        cli = _bare_cli()
        assert HermesCLI._tui_modal_active(cli) is False

    def test_true_when_approval_panel(self):
        cli = _bare_cli()
        cli._approval_state = {"response_queue": None}
        assert HermesCLI._tui_modal_active(cli) is True


class TestWantSimpleApproval:
    def test_env_truthy(self, monkeypatch):
        cli = _bare_cli()
        monkeypatch.setenv("HERMES_TUI_SIMPLE_APPROVAL", "1")
        assert HermesCLI._want_simple_approval_prompt(cli) is True

    def test_config_display_flag(self, monkeypatch):
        cli = _bare_cli()
        monkeypatch.delenv("HERMES_TUI_SIMPLE_APPROVAL", raising=False)

        def _fake_load():
            return {"display": {"tui_simple_approval": True}}

        with patch("hermes_cli.config.load_config", _fake_load):
            assert HermesCLI._want_simple_approval_prompt(cli) is True


class TestApprovalSimpleDelegation:
    def test_delegates_to_plain_prompt(self, monkeypatch):
        cli = _bare_cli()
        cli._approval_lock = threading.Lock()
        cli._invalidate = MagicMock()

        monkeypatch.setenv("HERMES_TUI_SIMPLE_APPROVAL", "1")

        with patch("tools.approval.prompt_dangerous_approval", return_value="session") as mock_prompt:
            out = HermesCLI._approval_callback(cli, "rm -rf /", "bad idea", allow_permanent=True)

        assert out == "session"
        mock_prompt.assert_called_once()
        kwargs = mock_prompt.call_args.kwargs
        assert kwargs["approval_callback"] is None
        cli._invalidate.assert_called_once()
