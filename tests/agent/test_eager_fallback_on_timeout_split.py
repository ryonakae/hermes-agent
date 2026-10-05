"""Focused regression coverage for the split turn-loop eager-timeout fallback."""

from types import SimpleNamespace

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_iteration_prep import apply_retry_restarts
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState
from agent.agent_init import _apply_agent_section
from hermes_cli.config import load_config


class _Budget:
    def __init__(self):
        self.refunds = 0

    def refund(self):
        self.refunds += 1


class _Agent:
    provider = "custom"
    _credential_pool = None
    _fallback_index = 0
    _fallback_chain = [{"provider": "fallback", "model": "backup"}]
    _fallback_activated = False

    def __init__(self, eager):
        self._eager_fallback_on_timeout = eager
        self.statuses = []
        self.activated = []
        self.iteration_budget = _Budget()

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(kwargs)
        return True

    def _buffer_diagnostic_status(self, status):
        self.statuses.append(status)


def _route(agent, retry_count=1, reason=FailoverReason.timeout):
    retry = TurnRetryState()
    classified = ClassifiedError(reason=reason)
    verdict = route_classified_error(
        agent,
        RuntimeError("provider timed out"),
        classified,
        retry,
        error_msg="provider timed out",
        error_context={},
        recovered_with_pool=False,
        base_url="https://primary.example/v1",
        model="primary",
        messages=[{"role": "user", "content": "hello"}],
        api_messages=[{"role": "user", "content": "hello"}],
        system_message="system",
        active_system_prompt="system",
        conversation_history=[],
        retry_count=retry_count,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=1,
        api_call_count=1,
        effective_task_id=None,
    )
    return verdict, retry


def test_explicit_true_falls_back_on_first_timeout_and_arms_rebuild():
    agent = _Agent(True)

    verdict, retry = _route(agent)

    assert verdict.action == "break"
    assert verdict.retry_count == 0
    assert len(agent.activated) == 1
    assert retry.restart_with_rebuilt_messages is True

    restarted = apply_retry_restarts(
        agent,
        _retry=retry,
        response=None,
        interrupted=False,
        messages=[],
        conversation_history=[],
        user_message="hello",
        api_kwargs={},
        current_turn_user_idx=0,
        final_response=None,
        retry_count=0,
        max_retries=3,
        api_call_count=1,
        restart_count=0,
        length_continue_retries=0,
        _preflight_compression_blocked=True,
        _turn_exit_reason="unknown",
    )

    assert restarted.action == "continue"
    assert restarted._preflight_compression_blocked is False
    assert agent.iteration_budget.refunds == 1


def test_unset_or_false_keeps_timeout_on_upstream_retry_path():
    for eager in (False, None):
        agent = _Agent(eager)
        verdict, retry = _route(agent)
        assert verdict.action == "fallthrough"
        assert agent.activated == []
        assert retry.restart_with_rebuilt_messages is False


def test_eager_timeout_flag_does_not_change_other_transport_errors():
    agent = _Agent(True)

    verdict, retry = _route(agent, reason=FailoverReason.overloaded)

    assert verdict.action == "fallthrough"
    assert agent.activated == []
    assert retry.restart_with_rebuilt_messages is False


def test_agent_section_retains_explicit_eager_timeout_setting():
    agent = SimpleNamespace(run_budget_seconds=None)

    _apply_agent_section(agent, {"agent": {"eager_fallback_on_timeout": True}})
    assert agent._eager_fallback_on_timeout is True


def test_load_config_defaults_eager_timeout_flag_and_preserves_true(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"

    assert not config_path.exists()
    assert load_config()["agent"]["eager_fallback_on_timeout"] is False

    config_path.write_text("agent:\n  eager_fallback_on_timeout: true\n", encoding="utf-8")

    assert load_config()["agent"]["eager_fallback_on_timeout"] is True
