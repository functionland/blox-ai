"""Tests for src/runtime/tree_dsl.py — YAML loader + structural validator."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.runtime.tree_dsl import (
    EmitRecommendation,
    EmitVerdict,
    Tree,
    TreeValidationError,
    load_tree_registry,
    load_tree_yaml,
    parse_tree,
)


def _yaml(text: str) -> dict:
    import yaml
    return yaml.safe_load(textwrap.dedent(text))


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# parse_tree happy paths
# ---------------------------------------------------------------------------


def test_parse_minimal_tree():
    raw = _yaml("""
        id: minimal
        version: 1
        title: "Minimal tree"
        nodes:
          - id: node_a
            branches:
              - default:
                  stop: true
    """)
    tree = parse_tree(raw)
    assert tree.id == "minimal"
    assert tree.version == 1
    assert tree.title == "Minimal tree"
    assert tree.entry == "node_a"          # defaults to first node
    assert list(tree.nodes.keys()) == ["node_a"]
    branch = tree.nodes["node_a"].branches[0]
    assert branch.when is None              # default branch
    assert branch.then.stop is True


def test_parse_tree_with_emit_verdict_and_recommendation():
    raw = _yaml("""
        id: rich
        version: 2
        title: "Rich tree"
        nodes:
          - id: check
            diag: internet
            timeout_s: 8
            branches:
              - when: "result.ok == False"
                then:
                  emit_thought: "Internet down"
                  emit_verdict:
                    summary: "No internet"
                    severity: red
                    root_cause: no_internet
                  emit_recommendation:
                    action_name: restart_network
                    tier: 2
                    reasoning: "Bounce systemd-networkd"
                    confidence: 0.9
                    args:
                      service: NetworkManager
                  stop: true
              - default:
                  next: check
    """)
    tree = parse_tree(raw)
    node = tree.nodes["check"]
    assert node.diag == "internet"
    assert node.timeout_s == 8.0
    branch = node.branches[0]
    assert branch.when == "result.ok == False"
    assert branch.then.emit_thought == "Internet down"
    assert isinstance(branch.then.emit_verdict, EmitVerdict)
    assert branch.then.emit_verdict.severity == "red"
    assert isinstance(branch.then.emit_recommendation, EmitRecommendation)
    assert branch.then.emit_recommendation.action_name == "restart_network"
    assert branch.then.emit_recommendation.tier == 2
    assert branch.then.emit_recommendation.confidence == 0.9
    assert branch.then.emit_recommendation.args == {"service": "NetworkManager"}
    assert branch.then.stop is True


def test_parse_tree_with_explicit_entry():
    raw = _yaml("""
        id: explicit_entry
        version: 1
        title: "Explicit entry"
        entry: node_b
        nodes:
          - id: node_a
            branches: [{default: {stop: true}}]
          - id: node_b
            branches: [{default: {stop: true}}]
    """)
    tree = parse_tree(raw)
    assert tree.entry == "node_b"


def test_parse_tree_with_subtree():
    raw = _yaml("""
        id: with_subtree
        version: 1
        title: "Parent"
        nodes:
          - id: root
            branches:
              - default:
                  next: deep_check    # references subtree
        subtrees:
          deep_check:
            nodes:
              - id: inner
                branches: [{default: {stop: true}}]
    """)
    tree = parse_tree(raw)
    assert "deep_check" in tree.subtrees
    assert "inner" in tree.subtrees["deep_check"].nodes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_reject_missing_id():
    with pytest.raises(TreeValidationError, match="missing required key 'id'"):
        parse_tree({"version": 1, "title": "x", "nodes": [{"id": "a", "branches": [{"default": {}}]}]})


def test_reject_wrong_type_for_version():
    with pytest.raises(TreeValidationError, match="must be int"):
        parse_tree({"id": "x", "version": "one", "title": "x",
                    "nodes": [{"id": "a", "branches": [{"default": {}}]}]})


def test_reject_empty_nodes():
    with pytest.raises(TreeValidationError, match="nodes list is empty"):
        parse_tree({"id": "x", "version": 1, "title": "x", "nodes": []})


def test_reject_duplicate_node_ids():
    raw = _yaml("""
        id: dup
        version: 1
        title: x
        nodes:
          - id: same
            branches: [{default: {stop: true}}]
          - id: same
            branches: [{default: {stop: true}}]
    """)
    with pytest.raises(TreeValidationError, match="duplicate node id 'same'"):
        parse_tree(raw)


def test_reject_unknown_entry():
    raw = _yaml("""
        id: bad_entry
        version: 1
        title: x
        entry: nonexistent
        nodes:
          - id: real
            branches: [{default: {stop: true}}]
    """)
    with pytest.raises(TreeValidationError, match="entry node 'nonexistent' not in nodes"):
        parse_tree(raw)


def test_reject_branch_next_to_unknown_node():
    raw = _yaml("""
        id: bad_next
        version: 1
        title: x
        nodes:
          - id: only
            branches:
              - default:
                  next: nowhere
    """)
    with pytest.raises(TreeValidationError, match="references unknown node/subtree 'nowhere'"):
        parse_tree(raw)


def test_reject_branch_with_neither_when_nor_default():
    raw = _yaml("""
        id: bad
        version: 1
        title: x
        nodes:
          - id: only
            branches:
              - then: {stop: true}        # missing both when: and default:
    """)
    with pytest.raises(TreeValidationError, match="must have non-empty"):
        parse_tree(raw)


def test_reject_multiple_default_branches():
    raw = _yaml("""
        id: bad
        version: 1
        title: x
        nodes:
          - id: only
            branches:
              - default: {stop: true}
              - default: {stop: true}
    """)
    with pytest.raises(TreeValidationError, match="multiple default branches"):
        parse_tree(raw)


def test_reject_invalid_severity():
    raw = _yaml("""
        id: bad
        version: 1
        title: x
        nodes:
          - id: only
            branches:
              - when: "True"
                then:
                  emit_verdict: {summary: x, severity: pink, root_cause: y}
    """)
    with pytest.raises(TreeValidationError, match="severity 'pink'"):
        parse_tree(raw)


def test_reject_invalid_tier():
    raw = _yaml("""
        id: bad
        version: 1
        title: x
        nodes:
          - id: only
            branches:
              - when: "True"
                then:
                  emit_recommendation:
                    action_name: x
                    tier: 5
                    reasoning: y
    """)
    with pytest.raises(TreeValidationError, match="tier must be 1\\|2\\|3"):
        parse_tree(raw)


def test_reject_emit_verdict_missing_required_field():
    raw = _yaml("""
        id: bad
        version: 1
        title: x
        nodes:
          - id: only
            branches:
              - when: "True"
                then:
                  emit_verdict: {summary: x, severity: red}    # no root_cause
    """)
    with pytest.raises(TreeValidationError, match="emit_verdict missing required field"):
        parse_tree(raw)


# ---------------------------------------------------------------------------
# load_tree_yaml — file-level
# ---------------------------------------------------------------------------


def test_load_tree_yaml_happy(tmp_path):
    p = _write(tmp_path, "simple.yaml", """
        id: simple
        version: 1
        title: Simple
        nodes:
          - id: only
            branches: [{default: {stop: true}}]
    """)
    tree = load_tree_yaml(p)
    assert tree.id == "simple"


def test_load_tree_yaml_file_missing(tmp_path):
    with pytest.raises(TreeValidationError, match="cannot read"):
        load_tree_yaml(tmp_path / "nonexistent.yaml")


def test_load_tree_yaml_malformed(tmp_path):
    p = _write(tmp_path, "bad.yaml", """
        id: x
        version: 1
        title: x
        nodes:
          - this is: invalid YAML scalar where mapping expected
            because the next key has wrong indent
        - id: a
          branches: [{default: {stop: true}}]
    """)
    # Should raise EITHER TreeValidationError (structural) or yaml parse
    # error wrapped — either way load_tree_yaml MUST raise.
    with pytest.raises(TreeValidationError):
        load_tree_yaml(p)


def test_load_tree_yaml_not_a_mapping(tmp_path):
    p = _write(tmp_path, "list.yaml", "- 1\n- 2\n- 3\n")
    with pytest.raises(TreeValidationError, match="top-level must be a mapping"):
        load_tree_yaml(p)


# ---------------------------------------------------------------------------
# load_tree_registry — multi-file + cross-validation
# ---------------------------------------------------------------------------


def test_load_registry_validates_diag_tools(tmp_path):
    _write(tmp_path, "a.yaml", """
        id: a
        version: 1
        title: A
        nodes:
          - id: only
            diag: not_a_real_tool
            branches: [{default: {stop: true}}]
    """)
    with pytest.raises(TreeValidationError, match="references unknown diag tool 'not_a_real_tool'"):
        load_tree_registry(tmp_path, known_diag_tools={"internet"}, known_action_names=set())


def test_load_registry_validates_action_names(tmp_path):
    _write(tmp_path, "a.yaml", """
        id: a
        version: 1
        title: A
        nodes:
          - id: only
            branches:
              - when: "True"
                then:
                  emit_recommendation:
                    action_name: nuke_from_orbit
                    tier: 3
                    reasoning: dramatic
    """)
    with pytest.raises(TreeValidationError, match="references unknown action 'nuke_from_orbit'"):
        load_tree_registry(tmp_path, known_diag_tools=set(),
                            known_action_names={"docker.restart"})


def test_load_registry_validates_goto_tree(tmp_path):
    _write(tmp_path, "a.yaml", """
        id: a
        version: 1
        title: A
        nodes:
          - id: only
            branches:
              - when: "True"
                then:
                  goto_tree: nonexistent
    """)
    with pytest.raises(TreeValidationError, match="goto_tree references unknown tree 'nonexistent'"):
        load_tree_registry(tmp_path, known_diag_tools=set(), known_action_names=set())


def test_load_registry_happy_two_trees_with_cross_goto(tmp_path):
    _write(tmp_path, "a.yaml", """
        id: a
        version: 1
        title: A
        nodes:
          - id: a1
            branches:
              - when: "True"
                then:
                  goto_tree: b
    """)
    _write(tmp_path, "b.yaml", """
        id: b
        version: 1
        title: B
        nodes:
          - id: b1
            branches: [{default: {stop: true}}]
    """)
    registry = load_tree_registry(tmp_path, known_diag_tools=set(), known_action_names=set())
    assert set(registry.keys()) == {"a", "b"}


def test_load_registry_rejects_duplicate_tree_id(tmp_path):
    _write(tmp_path, "x.yaml", """
        id: same
        version: 1
        title: X
        nodes: [{id: only, branches: [{default: {stop: true}}]}]
    """)
    _write(tmp_path, "y.yaml", """
        id: same
        version: 1
        title: Y
        nodes: [{id: only, branches: [{default: {stop: true}}]}]
    """)
    with pytest.raises(TreeValidationError, match="duplicate tree id 'same'"):
        load_tree_registry(tmp_path, known_diag_tools=set(), known_action_names=set())


def test_load_registry_empty_dir(tmp_path):
    with pytest.raises(TreeValidationError, match="no \\*\\.yaml trees found"):
        load_tree_registry(tmp_path, known_diag_tools=set(), known_action_names=set())
