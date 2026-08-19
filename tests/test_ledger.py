"""Tests for session resumption bookkeeping.

Resuming is only safe if a mismatch is caught. Every test here is about the
gateway declining to resume when it cannot prove the Claude session holds the
conversation the client just sent - answering confidently from the wrong
context is far worse than paying to replay the transcript.
"""

from types import SimpleNamespace

from claude_code_api.utils.ledger import (
    MODE_FLATTEN,
    MODE_RESUME,
    SessionLedger,
    chain_hash,
    conversation_fingerprints,
    fingerprint_message,
    fingerprint_system,
    is_session_missing_error,
    match_prefix,
    plan_turn,
    render_delta,
)


def msg(role, content=None, tool_calls=None, tool_call_id=None, name=None):
    return SimpleNamespace(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
    )


def call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def ledger_for(messages, names=None, system="sys"):
    ledger = SessionLedger(system_fingerprint=fingerprint_system(system))
    ledger.extend(messages, names or {})
    return ledger


class TestFingerprinting:
    def test_identical_messages_match(self):
        assert fingerprint_message(msg("user", "hi")) == fingerprint_message(
            msg("user", "hi")
        )

    def test_role_is_part_of_identity(self):
        assert fingerprint_message(msg("user", "hi")) != fingerprint_message(
            msg("assistant", "hi")
        )

    def test_surrounding_whitespace_is_ignored(self):
        assert fingerprint_message(msg("user", " hi ")) == fingerprint_message(
            msg("user", "hi")
        )

    def test_argument_key_order_does_not_matter(self):
        """A client re-serialising our tool call will not preserve byte order."""
        ours = msg("assistant", None, tool_calls=[call("c1", "t", '{"a":1,"b":2}')])
        theirs = msg(
            "assistant", None, tool_calls=[call("c1", "t", '{"b": 2, "a": 1}')]
        )
        assert fingerprint_message(ours) == fingerprint_message(theirs)

    def test_arguments_as_an_object_match_the_string_form(self):
        as_string = msg("assistant", None, tool_calls=[call("c1", "t", '{"a":1}')])
        as_object = msg("assistant", None, tool_calls=[call("c1", "t", {"a": 1})])
        assert fingerprint_message(as_string) == fingerprint_message(as_object)

    def test_empty_and_null_content_match(self):
        """Clients routinely drop `content: null` from an assistant tool call."""
        calls = [call("c1", "t", "{}")]
        assert fingerprint_message(
            msg("assistant", None, tool_calls=calls)
        ) == fingerprint_message(msg("assistant", "", tool_calls=calls))

    def test_tool_results_are_distinguished_by_call_id(self):
        a = msg("tool", "same text", tool_call_id="c1")
        b = msg("tool", "same text", tool_call_id="c2")
        assert fingerprint_message(a) != fingerprint_message(b)

    def test_different_arguments_differ(self):
        a = msg("assistant", None, tool_calls=[call("c1", "t", '{"city":"Paris"}')])
        b = msg("assistant", None, tool_calls=[call("c1", "t", '{"city":"Berlin"}')])
        assert fingerprint_message(a) != fingerprint_message(b)


class TestConversationIndex:
    def test_longest_known_prefix_wins(self):
        fingerprints = ["a", "b", "c", "d"]
        index = {
            chain_hash(fingerprints[:2]): "sess-1",
            chain_hash(fingerprints[:4]): "sess-1",
        }
        assert match_prefix(index, fingerprints) == ("sess-1", 4)

    def test_shorter_prefix_matches_when_it_is_all_there_is(self):
        fingerprints = ["a", "b", "c"]
        index = {chain_hash(fingerprints[:2]): "sess-1"}
        assert match_prefix(index, fingerprints) == ("sess-1", 2)

    def test_unknown_conversation_matches_nothing(self):
        index = {chain_hash(["a", "b"]): "sess-1"}
        assert match_prefix(index, ["x", "y"]) is None

    def test_a_changed_opening_message_breaks_the_chain(self):
        """Editing history must not silently continue the old session."""
        original = conversation_fingerprints([msg("user", "about Paris")])
        index = {chain_hash(original): "sess-1"}
        edited = conversation_fingerprints([msg("user", "about Berlin")])
        assert match_prefix(index, edited) is None

    def test_system_messages_are_not_part_of_the_chain(self):
        with_system = conversation_fingerprints([msg("system", "s"), msg("user", "hi")])
        without = conversation_fingerprints([msg("user", "hi")])
        assert with_system == without


class TestPlanTurn:
    def _base(self):
        return [msg("system", "sys"), msg("user", "first question")]

    def test_resumes_with_only_the_new_message(self):
        history = [msg("user", "first question"), msg("assistant", "first answer")]
        ledger = ledger_for(history)
        messages = self._base() + [
            msg("assistant", "first answer"),
            msg("user", "second question"),
        ]

        plan = plan_turn(messages, ledger, "sdk-1", "sys")

        assert plan.mode == MODE_RESUME
        assert plan.resume_session_id == "sdk-1"
        assert plan.prompt == "second question"
        assert len(plan.consumed) == 1

    def test_tool_results_are_labelled_for_the_model(self):
        history = [
            msg("user", "weather?"),
            msg("assistant", None, tool_calls=[call("c1", "get_weather", "{}")]),
        ]
        ledger = ledger_for(history, {"c1": "get_weather"})
        messages = (
            [msg("system", "sys")]
            + history
            + [msg("tool", '{"temp_c":18}', tool_call_id="c1")]
        )

        plan = plan_turn(messages, ledger, "sdk-1", "sys")

        assert plan.mode == MODE_RESUME
        assert "get_weather" in plan.prompt
        assert "c1" in plan.prompt
        assert "temp_c" in plan.prompt

    def test_flattens_without_a_claude_session(self):
        ledger = ledger_for([msg("user", "first question")])
        plan = plan_turn(self._base(), ledger, None, "sys")

        assert plan.mode == MODE_FLATTEN
        assert "no Claude session" in plan.reason

    def test_flattens_with_no_ledger(self):
        plan = plan_turn(self._base(), None, "sdk-1", "sys")
        assert plan.mode == MODE_FLATTEN

    def test_flattens_when_the_system_prompt_changed(self):
        """Claude's system prompt is fixed at session start and cannot be retracted."""
        ledger = ledger_for([msg("user", "first question")], system="sys")
        messages = self._base() + [msg("user", "next")]

        plan = plan_turn(messages, ledger, "sdk-1", "a different prompt")

        assert plan.mode == MODE_FLATTEN
        assert "system prompt changed" in plan.reason

    def test_flattens_when_the_client_edited_history(self):
        ledger = ledger_for(
            [msg("user", "first question"), msg("assistant", "first answer")]
        )
        messages = [
            msg("system", "sys"),
            msg("user", "first question EDITED"),
            msg("assistant", "first answer"),
            msg("user", "next"),
        ]

        plan = plan_turn(messages, ledger, "sdk-1", "sys")

        assert plan.mode == MODE_FLATTEN
        assert "diverges" in plan.reason

    def test_flattens_when_the_client_truncated_history(self):
        ledger = ledger_for(
            [
                msg("user", "first question"),
                msg("assistant", "first answer"),
                msg("user", "second question"),
            ]
        )
        messages = [msg("system", "sys"), msg("user", "first question")]

        plan = plan_turn(messages, ledger, "sdk-1", "sys")
        assert plan.mode == MODE_FLATTEN

    def test_flattens_when_there_is_nothing_new(self):
        history = [msg("user", "first question")]
        ledger = ledger_for(history)

        plan = plan_turn(self._base(), ledger, "sdk-1", "sys")

        assert plan.mode == MODE_FLATTEN
        assert "nothing new" in plan.reason

    def test_flattens_when_the_delta_holds_an_unknown_assistant_turn(self):
        """An assistant message the session never produced means the ledger is stale."""
        ledger = ledger_for([msg("user", "first question")])
        messages = self._base() + [
            msg("assistant", "a reply we never made"),
            msg("user", "next"),
        ]

        plan = plan_turn(messages, ledger, "sdk-1", "sys")

        assert plan.mode == MODE_FLATTEN
        assert "assistant message" in plan.reason

    def test_flatten_plans_still_carry_a_usable_prompt(self):
        ledger = ledger_for([msg("user", "x")])
        messages = self._base() + [msg("assistant", "a"), msg("user", "b")]

        plan = plan_turn(messages, ledger, "sdk-1", "sys")

        assert plan.mode == MODE_FLATTEN
        assert "first question" in plan.prompt
        assert "b" in plan.prompt


class TestRenderDelta:
    def test_plain_user_message(self):
        assert render_delta([msg("user", "hello")], {}) == "hello"

    def test_tool_result_names_come_from_the_ledger(self):
        rendered = render_delta(
            [msg("tool", "result body", tool_call_id="c1")], {"c1": "search_nodes"}
        )
        assert "search_nodes" in rendered
        assert "result body" in rendered

    def test_unknown_call_id_still_renders(self):
        rendered = render_delta([msg("tool", "body", tool_call_id="zzz")], {})
        assert "body" in rendered

    def test_parallel_results_are_all_present(self):
        rendered = render_delta(
            [
                msg("tool", "first", tool_call_id="c1"),
                msg("tool", "second", tool_call_id="c2"),
            ],
            {"c1": "a", "c2": "b"},
        )
        assert "first" in rendered and "second" in rendered


class TestSessionMissingDetection:
    def test_recognises_the_sdk_message(self):
        error = RuntimeError(
            "Claude Code returned an error result: No conversation found with "
            "session ID: 0000 (exit code: 1)"
        )
        assert is_session_missing_error(error) is True

    def test_other_failures_are_not_mistaken_for_it(self):
        assert is_session_missing_error(RuntimeError("rate limited")) is False
        assert is_session_missing_error(RuntimeError("boom")) is False


class TestSessionManagerLedger:
    def _manager(self):
        from claude_code_api.core.session_manager import SessionManager

        manager = SessionManager.__new__(SessionManager)
        manager.active_sessions = {}
        manager.cli_session_index = {}
        manager.ledgers = {}
        manager.conversation_index = {}
        return manager

    def test_recording_a_turn_makes_it_findable_by_content(self):
        manager = self._manager()
        conversation = [msg("user", "hi"), msg("assistant", "hello")]
        manager.record_turn("sess-A", conversation, {})

        later = conversation_fingerprints(conversation + [msg("user", "again")])
        assert manager.find_resumable(later) == ("sess-A", 2)

    def test_unknown_conversation_is_not_found(self):
        manager = self._manager()
        manager.record_turn("sess-A", [msg("user", "hi")], {})
        assert (
            manager.find_resumable(conversation_fingerprints([msg("user", "x")]))
            is None
        )

    def test_discarding_removes_it_from_the_index(self):
        manager = self._manager()
        conversation = [msg("user", "hi")]
        manager.record_turn("sess-A", conversation, {})
        manager.discard_ledger("sess-A")

        assert manager.find_resumable(conversation_fingerprints(conversation)) is None
        assert manager.get_ledger("sess-A") is None

    def test_a_missing_ledger_invalidates_an_index_hit(self):
        """Belt and braces: never resume against a session we cannot verify."""
        manager = self._manager()
        conversation = [msg("user", "hi")]
        manager.record_turn("sess-A", conversation, {})
        manager.ledgers.pop("sess-A")

        assert manager.find_resumable(conversation_fingerprints(conversation)) is None
