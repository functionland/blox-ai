"""Tests for src/runtime/tree_runner.py — walks trees + emits SSE events."""
from __future__ import annotations

import asyncio

import pytest

from src.runtime.tree_dsl import parse_tree
from src.runtime.tree_runner import (
    MAX_NODES_PER_RUN,
    MAX_TREE_HOPS,
    TreeRunner,
)


def _yaml(text: str) -> dict:
    import textwrap
    import yaml
    return yaml.safe_load(textwrap.dedent(text))


def _build_runner(tree_yamls: dict[str, str], diag_results: dict | None = None):
    """Build a TreeRunner with trees parsed from inline YAML + a mock
    diag executor returning canned per-tool results."""
    trees = {tid: parse_tree(_yaml(text)) for tid, text in tree_yamls.items()}
    diag_results = diag_results or {}

    async def fake_executor(tool: str, args: dict):
        if tool in diag_results:
            v = diag_results[tool]
            return v() if callable(v) else v
        return {}

    return TreeRunner(trees=trees, tool_executor=fake_executor)


async def _collect(generator):
    return [ev async for ev in generator]


# ---------------------------------------------------------------------------
# Single-node trees
# ---------------------------------------------------------------------------


def test_run_single_node_with_default_stop():
    runner = _build_runner({
        "x": """
            id: x
            version: 1
            title: x
            nodes:
              - id: only
                branches: [{default: {stop: true}}]
        """,
    })
    events = asyncio.run(_collect(runner.run("x")))
    # No verdict emitted by the tree → runner synthesizes one.
    assert events[-1]["type"] == "verdict"
    assert events[-1]["payload"]["root_cause"] == "tree_indeterminate"


def test_run_single_node_emits_verdict():
    runner = _build_runner({
        "x": """
            id: x
            version: 1
            title: x
            nodes:
              - id: only
                branches:
                  - default:
                      emit_verdict:
                        summary: All good
                        severity: green
                        root_cause: nominal
                      stop: true
        """,
    })
    events = asyncio.run(_collect(runner.run("x")))
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["payload"]["root_cause"] == "nominal"


def test_run_unknown_scenario_emits_error():
    runner = _build_runner({
        "x": """
            id: x
            version: 1
            title: x
            nodes:
              - id: only
                branches: [{default: {stop: true}}]
        """,
    })
    events = asyncio.run(_collect(runner.run("not_a_scenario")))
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "unknown_scenario"
    assert events[0]["recoverable"] is False


# ---------------------------------------------------------------------------
# diag tool call + branch matching
# ---------------------------------------------------------------------------


def test_diag_called_and_result_branches_correctly():
    runner = _build_runner(
        {
            "internet_check": """
                id: internet_check
                version: 1
                title: x
                nodes:
                  - id: probe
                    diag: internet
                    branches:
                      - when: "result.ok == False"
                        then:
                          emit_verdict:
                            summary: No internet
                            severity: red
                            root_cause: no_internet
                          stop: true
                      - default:
                          emit_verdict:
                            summary: All good
                            severity: green
                            root_cause: ok
                          stop: true
            """,
        },
        diag_results={"diag/internet": {"ok": False}},
    )
    events = asyncio.run(_collect(runner.run("internet_check")))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "verdict" in types
    verdict = [e for e in events if e["type"] == "verdict"][0]
    assert verdict["payload"]["root_cause"] == "no_internet"


def test_diag_result_cached_within_run():
    """A second node referencing the same diag should NOT re-execute."""
    call_count = {"n": 0}

    async def counting_executor(tool: str, args: dict):
        call_count["n"] += 1
        return {"ok": True}

    runner = TreeRunner(
        trees={
            "cache_test": parse_tree(_yaml("""
                id: cache_test
                version: 1
                title: x
                nodes:
                  - id: first
                    diag: internet
                    branches: [{default: {next: second}}]
                  - id: second
                    diag: internet
                    branches:
                      - default:
                          emit_verdict:
                            summary: x
                            severity: green
                            root_cause: ok
                          stop: true
            """)),
        },
        tool_executor=counting_executor,
    )
    asyncio.run(_collect(runner.run("cache_test")))
    assert call_count["n"] == 1, "diag should hit cache on second use"


def test_diag_timeout_emits_failed_tool_result():
    async def slow_executor(tool: str, args: dict):
        await asyncio.sleep(10)
        return {}

    runner = TreeRunner(
        trees={
            "to": parse_tree(_yaml("""
                id: to
                version: 1
                title: x
                nodes:
                  - id: probe
                    diag: internet
                    timeout_s: 0.05
                    branches: [{default: {stop: true}}]
            """)),
        },
        tool_executor=slow_executor,
    )
    events = asyncio.run(_collect(runner.run("to")))
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["ok"] is False
    assert "timed out" in tool_results[0]["error"]


def test_diag_exception_emits_failed_tool_result():
    async def raising(tool: str, args: dict):
        raise RuntimeError("simulated impl crash")

    runner = TreeRunner(
        trees={
            "ex": parse_tree(_yaml("""
                id: ex
                version: 1
                title: x
                nodes:
                  - id: probe
                    diag: internet
                    branches: [{default: {stop: true}}]
            """)),
        },
        tool_executor=raising,
    )
    events = asyncio.run(_collect(runner.run("ex")))
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["ok"] is False
    assert "RuntimeError" in tool_results[0]["error"]


# ---------------------------------------------------------------------------
# Multi-node cascade + branch ordering
# ---------------------------------------------------------------------------


def test_cascade_first_matching_branch_wins():
    runner = _build_runner(
        {
            "cascade": """
                id: cascade
                version: 1
                title: x
                nodes:
                  - id: a
                    diag: internet
                    branches:
                      - when: "result.x == 1"
                        then:
                          emit_thought: matched-a
                          next: b
                      - when: "result.x == 1"   # would also match, but first wins
                        then:
                          emit_thought: wrong
                          stop: true
                  - id: b
                    branches:
                      - default:
                          emit_verdict:
                            summary: arrived at b
                            severity: green
                            root_cause: ok
                          stop: true
            """,
        },
        diag_results={"diag/internet": {"x": 1}},
    )
    events = asyncio.run(_collect(runner.run("cascade")))
    thoughts = [e for e in events if e["type"] == "thought"]
    assert any("matched-a" in t["payload"] for t in thoughts)
    assert not any("wrong" in t["payload"] for t in thoughts)


def test_no_match_no_default_halts_cleanly():
    runner = _build_runner(
        {
            "nomatch": """
                id: nomatch
                version: 1
                title: x
                nodes:
                  - id: only
                    diag: internet
                    branches:
                      - when: "result.x == 999"
                        then:
                          emit_verdict:
                            summary: should not appear
                            severity: red
                            root_cause: nope
                          stop: true
            """,
        },
        diag_results={"diag/internet": {"x": 1}},
    )
    events = asyncio.run(_collect(runner.run("nomatch")))
    # No branch matched, no default → runner falls off; synth verdict.
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["payload"]["root_cause"] == "tree_indeterminate"


# ---------------------------------------------------------------------------
# Sub-trees + goto_tree
# ---------------------------------------------------------------------------


def test_subtree_dispatch_inline():
    runner = _build_runner({
        "with_sub": """
            id: with_sub
            version: 1
            title: x
            nodes:
              - id: root
                branches:
                  - default: {next: deep_check}
            subtrees:
              deep_check:
                nodes:
                  - id: inner
                    branches:
                      - default:
                          emit_verdict:
                            summary: from-subtree
                            severity: yellow
                            root_cause: subtree_finished
                          stop: true
        """,
    })
    events = asyncio.run(_collect(runner.run("with_sub")))
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert verdicts[-1]["payload"]["root_cause"] == "subtree_finished"


def test_goto_tree_cross_tree_redirect():
    runner = _build_runner({
        "first": """
            id: first
            version: 1
            title: x
            nodes:
              - id: a
                branches:
                  - default: {goto_tree: second}
        """,
        "second": """
            id: second
            version: 1
            title: y
            nodes:
              - id: only
                branches:
                  - default:
                      emit_verdict:
                        summary: from-second-tree
                        severity: green
                        root_cause: redirected
                      stop: true
        """,
    })
    events = asyncio.run(_collect(runner.run("first")))
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert verdicts[-1]["payload"]["root_cause"] == "redirected"


def test_goto_tree_to_unknown_tree_emits_error():
    runner = _build_runner({
        "first": """
            id: first
            version: 1
            title: x
            nodes:
              - id: a
                branches:
                  - default:
                      goto_tree: nonexistent
        """,
    })
    events = asyncio.run(_collect(runner.run("first")))
    errors = [e for e in events if e["type"] == "error"]
    assert any(e["code"] == "unknown_goto_tree" for e in errors)


# ---------------------------------------------------------------------------
# Recommendation emission
# ---------------------------------------------------------------------------


def test_recommendation_event_has_required_fields():
    runner = _build_runner({
        "rec": """
            id: rec
            version: 1
            title: x
            nodes:
              - id: only
                branches:
                  - default:
                      emit_recommendation:
                        action_name: docker.restart
                        tier: 2
                        reasoning: bounce the container
                        args: {container: ipfs_host}
                      emit_verdict:
                        summary: x
                        severity: yellow
                        root_cause: needs_restart
                      stop: true
        """,
    })
    events = asyncio.run(_collect(runner.run("rec")))
    recs = [e for e in events if e["type"] == "recommended_action"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["action_name"] == "docker.restart"
    assert rec["tier"] == 2
    assert rec["args"] == {"container": "ipfs_host"}
    assert len(rec["approval_token"]) >= 64       # schema minLength=64
    assert 0 <= rec["confidence"] <= 1


# ---------------------------------------------------------------------------
# Defensive caps
# ---------------------------------------------------------------------------


def test_node_limit_caps_runaway_cycle():
    """Tree with a deliberate cycle MUST halt at MAX_NODES_PER_RUN
    rather than hanging the session."""
    runner = _build_runner({
        "cyc": """
            id: cyc
            version: 1
            title: x
            nodes:
              - id: a
                branches: [{default: {next: b}}]
              - id: b
                branches: [{default: {next: a}}]
        """,
    })
    events = asyncio.run(_collect(runner.run("cyc")))
    errors = [e for e in events if e["type"] == "error"]
    assert any(e["code"] == "tree_node_limit" for e in errors)


def test_tree_hop_limit_caps_cross_tree_cycle():
    """a → goto b → goto a → goto b → ... MUST halt."""
    trees = {}
    for tid, other in [("a", "b"), ("b", "a")]:
        trees[tid] = f"""
            id: {tid}
            version: 1
            title: x
            nodes:
              - id: only
                branches:
                  - default: {{goto_tree: {other}}}
        """
    runner = _build_runner(trees)
    events = asyncio.run(_collect(runner.run("a")))
    errors = [e for e in events if e["type"] == "error"]
    assert any(e["code"] == "tree_hop_limit" for e in errors)


def test_runner_catches_internal_exception():
    """If something inside the runner blows up unexpectedly, the run
    must emit a recoverable error event (not crash the session)."""
    bad_runner = TreeRunner(trees={}, tool_executor=None)
    # Smuggle in a bad tree post-construction to force an unexpected
    # error path.
    bad_runner.trees["broken"] = "not a Tree object"
    events = asyncio.run(_collect(bad_runner.run("broken")))
    errors = [e for e in events if e["type"] == "error"]
    assert any(e["code"] == "tree_runner_internal_error" for e in errors)
    assert all(e.get("recoverable") is True for e in errors)
