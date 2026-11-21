"""Only the LEGACY ``custom_providers:`` list rewrites a pinned ``api_mode``.

Two payloads reach the settings panel, and they do NOT share resolution:

* ``providers.<key>`` (source ``"providers"``) resolves through
  ``_custom_runtime``, which passes ``api_mode`` straight into the runtime dict —
  ``_resolve_plain_custom_api_mode``'s docstring says it plainly: "named custom
  providers opt in via ``api_mode``". A pinned ``codex_responses`` is honoured.
* the legacy ``custom_providers:`` list (source ``"custom_providers"``) goes
  through ``_resolve_plain_custom_api_mode``, which discards ``codex_responses``
  for any host that is not a known OpenAI endpoint and falls back to URL
  detection.

``_runtime_api_mode`` therefore keys the drop on ``source`` and never on a URL
guess. Flagging a named entry would warn about a mode the runtime is in fact
using — a wrong warning is worse than the silence it replaced.
"""

from hermes_cli.web_routers.config_env import _endpoint_row, _runtime_api_mode

RELAY = "https://api.example-relay.test/v1"


def _row(entry, base_url=RELAY, source="providers"):
    return _endpoint_row(
        "relay", "Relay", base_url, "some-model", ["some-model"], None, True, entry, True, source
    )


def test_named_provider_keeps_codex_responses():
    """``providers.<key>`` honours the pin, so nothing is reported as dropped."""
    effective, dropped = _runtime_api_mode({"api_mode": "codex_responses"}, RELAY, "providers")

    assert dropped == ""
    assert effective == "codex_responses"


def test_legacy_list_drops_codex_responses_on_a_non_openai_host():
    """The one path that really rewrites the mode."""
    effective, dropped = _runtime_api_mode({"api_mode": "codex_responses"}, RELAY, "custom_providers")

    assert dropped == "codex_responses"
    assert effective == "chat_completions", "URL detection yields the generic wire, never Responses"


def test_legacy_list_honours_codex_responses_on_openai_itself():
    """The rewrite is host-scoped; OpenAI's own endpoint keeps the mode."""
    effective, dropped = _runtime_api_mode(
        {"api_mode": "codex_responses"}, "https://api.openai.com/v1", "custom_providers"
    )

    assert dropped == ""
    assert effective == "codex_responses"


def test_chat_completions_never_reports_a_drop():
    """The mode the runtime already defaults to cannot be discarded, on either path."""
    for source in ("providers", "custom_providers"):
        assert _runtime_api_mode({"api_mode": "chat_completions"}, RELAY, source) == (
            "chat_completions",
            "",
        )


def test_anthropic_messages_survives_on_a_relay():
    """Only ``codex_responses`` is host-gated; Anthropic Messages is not."""
    effective, dropped = _runtime_api_mode({"api_mode": "anthropic_messages"}, RELAY, "custom_providers")

    assert dropped == ""
    assert effective == "anthropic_messages"


def test_unpinned_mode_reports_nothing():
    """``api_mode: ''`` means auto-detect: no configured value to compare against."""
    assert _runtime_api_mode({}, RELAY, "custom_providers") == ("", "")


def test_named_row_never_flags_the_pin():
    """The endpoint row (what the Desktop card reads) must stay quiet here."""
    row = _row({"api_mode": "codex_responses"})

    assert row["api_mode"] == "codex_responses"
    assert row["effective_api_mode"] == "codex_responses"
    assert row["dropped_api_mode"] == ""


def test_legacy_row_flags_the_rewrite_and_keeps_the_saved_value():
    """The card can still name both values on the path that does rewrite."""
    row = _row({"api_mode": "codex_responses"}, source="custom_providers")

    assert row["api_mode"] == "codex_responses", "the saved value stays visible"
    assert row["effective_api_mode"] == "chat_completions"
    assert row["dropped_api_mode"] == "codex_responses"


def test_direct_config_row_honours_its_pin():
    """A bare ``provider: custom`` slot is not the legacy list either."""
    effective, dropped = _runtime_api_mode({"api_mode": "codex_responses"}, RELAY, "direct-config")

    assert (effective, dropped) == ("codex_responses", "")
