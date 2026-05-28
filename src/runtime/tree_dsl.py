"""Tree DSL — YAML loader + schema validator for deterministic
troubleshooting trees.

YAML shape (informal; full grammar enforced by `Tree.validate`):

    id: disconnected
    version: 1
    title: "App shows Blox as disconnected"
    entry: internet_check          # optional; defaults to first node
    nodes:
      - id: internet_check
        diag: internet             # which diag/* tool to call
        timeout_s: 8               # optional, default 5.0
        branches:
          - when: "result.https_discovery_ok == False"
            then:
              emit_thought: "Internet to discovery failing..."
              next: dns_check
          - when: "result.captive_portal_likely == True"
            then:
              emit_verdict:
                summary: "Captive portal blocking discovery."
                severity: red
                root_cause: captive_portal
              stop: true
          - default:
              next: relay_check
    subtrees:                      # optional, named sub-trees in this file
      kubo_deep_check:
        entry: kubo_id_check
        nodes: [...]

Design constraints:
  - Trees are YAML in fula-ota repo; bind-mounted into container at
    /etc/fula/blox-ai/trees/. OTA-deployable (no container rebuild).
  - Schema validation at LOAD time — refuse to start the container if
    a tree file is malformed OR if a tree references a diag tool not
    in the known set OR an action_name not in action_whitelist.json.
  - Expressions parsed at load time too (simpleeval `parse()`); a
    syntactic error in `when:` is a load-time failure, not a
    runtime failure mid-troubleshoot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


logger = logging.getLogger("blox-ai.tree-dsl")


class TreeValidationError(ValueError):
    """Raised when a tree YAML is structurally invalid or references
    unknown diag tools / actions / node ids."""


@dataclass
class EmitVerdict:
    summary: str
    severity: str             # "green" | "yellow" | "red"
    root_cause: str


@dataclass
class EmitRecommendation:
    action_name: str
    tier: int                 # 1 | 2 | 3
    reasoning: str
    confidence: float = 1.0   # deterministic trees default to 1.0
    args: dict = field(default_factory=dict)
    expected_duration_s: Optional[float] = None


@dataclass
class Then:
    """Right-hand side of a branch: what to emit + where to go next.

    Three control-flow primitives for cross-/sub-tree composition:
      - next: jump to another node in this tree (or to a named subtree).
        Local-only; doesn't cross tree boundaries.
      - goto_tree: REDIRECT to a different top-level tree's entry.
        Caller's cascade ENDS unconditionally after the target tree
        finishes. Use when classifier guessed wrong and we want to
        redirect to the right scenario.
      - include_tree: CALL a different top-level tree's entry, then
        come BACK. If the called tree emitted a verdict, caller's
        cascade halts (we have a definitive answer). If the called
        tree did NOT emit a verdict, caller continues with `next:`.
        Use to compose hierarchies: not-earning INCLUDES disconnected;
        if disconnected found the problem, stop; else continue with
        cluster + pool checks.
    """
    emit_thought: Optional[str] = None
    emit_verdict: Optional[EmitVerdict] = None
    emit_recommendation: Optional[EmitRecommendation] = None
    next: Optional[str] = None           # next node id (or subtree name)
    goto_tree: Optional[str] = None      # cross-tree redirect (one-way)
    include_tree: Optional[str] = None   # cross-tree call (returns if no verdict)
    stop: bool = False                   # halt cascade


@dataclass
class Branch:
    when: Optional[str]   # None when this is the default branch
    then: Then


@dataclass
class Node:
    id: str
    branches: list[Branch]
    diag: Optional[str] = None        # None for compute-only nodes
    diag_args: dict = field(default_factory=dict)
    timeout_s: float = 5.0


@dataclass
class Tree:
    id: str
    version: int
    title: str
    nodes: dict[str, Node]                  # id → Node
    entry: str                              # first node to walk
    subtrees: dict[str, "Tree"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_tree_yaml(path: Union[str, Path]) -> Tree:
    """Load + validate a single tree YAML file. Raises
    TreeValidationError on any structural issue."""
    p = Path(path)
    try:
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except OSError as e:
        raise TreeValidationError(f"cannot read {p}: {e}") from e
    except yaml.YAMLError as e:
        raise TreeValidationError(f"YAML parse error in {p}: {e}") from e
    if not isinstance(raw, dict):
        raise TreeValidationError(f"{p}: top-level must be a mapping")
    return parse_tree(raw)


def parse_tree(raw: dict) -> Tree:
    """Build a Tree from a parsed dict. Pure (no I/O); call from
    load_tree_yaml or directly from tests."""
    _require(raw, "id", str)
    _require(raw, "version", int)
    _require(raw, "title", str)
    nodes_raw = _require(raw, "nodes", list)

    if not nodes_raw:
        raise TreeValidationError(f"tree {raw['id']!r}: nodes list is empty")

    nodes: dict[str, Node] = {}
    for n in nodes_raw:
        node = _parse_node(n, raw["id"])
        if node.id in nodes:
            raise TreeValidationError(
                f"tree {raw['id']!r}: duplicate node id {node.id!r}"
            )
        nodes[node.id] = node

    entry = raw.get("entry") or nodes_raw[0]["id"]
    if entry not in nodes:
        raise TreeValidationError(
            f"tree {raw['id']!r}: entry node {entry!r} not in nodes"
        )

    subtrees: dict[str, Tree] = {}
    for sub_id, sub_raw in (raw.get("subtrees") or {}).items():
        if not isinstance(sub_raw, dict):
            raise TreeValidationError(
                f"tree {raw['id']!r}: subtree {sub_id!r} must be a mapping"
            )
        # Inherit id+version+title from parent if missing — sub-trees
        # are conceptually private to the parent.
        sub_raw.setdefault("id", f"{raw['id']}.{sub_id}")
        sub_raw.setdefault("version", raw["version"])
        sub_raw.setdefault("title", f"{raw['title']} / {sub_id}")
        subtrees[sub_id] = parse_tree(sub_raw)

    # Cross-reference: every node's `next` must point to a known node
    # OR a known subtree name. `goto_tree` is checked at the multi-tree
    # validate_registry step (we don't know peer trees here).
    valid_next_ids = set(nodes.keys()) | set(subtrees.keys())
    for node in nodes.values():
        for branch in node.branches:
            target = branch.then.next
            if target is not None and target not in valid_next_ids:
                raise TreeValidationError(
                    f"tree {raw['id']!r} node {node.id!r}: "
                    f"branch.next references unknown node/subtree {target!r}"
                )

    return Tree(
        id=raw["id"],
        version=raw["version"],
        title=raw["title"],
        nodes=nodes,
        entry=entry,
        subtrees=subtrees,
    )


def _parse_node(raw: Any, tree_id: str) -> Node:
    if not isinstance(raw, dict):
        raise TreeValidationError(f"tree {tree_id!r}: node must be a mapping")
    _require(raw, "id", str)
    branches_raw = _require(raw, "branches", list)
    if not branches_raw:
        raise TreeValidationError(
            f"tree {tree_id!r} node {raw['id']!r}: branches list is empty"
        )
    branches = [_parse_branch(b, tree_id, raw["id"]) for b in branches_raw]
    # Exactly one default branch allowed (the catch-all).
    defaults = [b for b in branches if b.when is None]
    if len(defaults) > 1:
        raise TreeValidationError(
            f"tree {tree_id!r} node {raw['id']!r}: multiple default branches"
        )
    return Node(
        id=raw["id"],
        diag=raw.get("diag") if isinstance(raw.get("diag"), str) else None,
        diag_args=raw.get("diag_args") if isinstance(raw.get("diag_args"), dict) else {},
        timeout_s=float(raw.get("timeout_s", 5.0)),
        branches=branches,
    )


def _parse_branch(raw: Any, tree_id: str, node_id: str) -> Branch:
    if not isinstance(raw, dict):
        raise TreeValidationError(
            f"tree {tree_id!r} node {node_id!r}: branch must be a mapping"
        )
    if "default" in raw:
        return Branch(when=None, then=_parse_then(raw["default"], tree_id, node_id))
    when = raw.get("when")
    if not isinstance(when, str) or not when.strip():
        raise TreeValidationError(
            f"tree {tree_id!r} node {node_id!r}: "
            f"branch must have non-empty `when:` OR be a `default:`"
        )
    then_raw = raw.get("then")
    if not isinstance(then_raw, dict):
        raise TreeValidationError(
            f"tree {tree_id!r} node {node_id!r}: branch.then must be a mapping"
        )
    return Branch(when=when.strip(), then=_parse_then(then_raw, tree_id, node_id))


def _parse_then(raw: Any, tree_id: str, node_id: str) -> Then:
    if not isinstance(raw, dict):
        raise TreeValidationError(
            f"tree {tree_id!r} node {node_id!r}: then-block must be a mapping"
        )
    emit_verdict = None
    if "emit_verdict" in raw:
        v = raw["emit_verdict"]
        if not isinstance(v, dict):
            raise TreeValidationError(
                f"tree {tree_id!r} node {node_id!r}: emit_verdict must be a mapping"
            )
        try:
            emit_verdict = EmitVerdict(
                summary=str(v["summary"]),
                severity=str(v["severity"]),
                root_cause=str(v["root_cause"]),
            )
        except KeyError as e:
            raise TreeValidationError(
                f"tree {tree_id!r} node {node_id!r}: "
                f"emit_verdict missing required field {e}"
            ) from e
        if emit_verdict.severity not in ("green", "yellow", "red"):
            raise TreeValidationError(
                f"tree {tree_id!r} node {node_id!r}: "
                f"severity {emit_verdict.severity!r} not in green|yellow|red"
            )

    emit_rec = None
    if "emit_recommendation" in raw:
        r = raw["emit_recommendation"]
        if not isinstance(r, dict):
            raise TreeValidationError(
                f"tree {tree_id!r} node {node_id!r}: emit_recommendation must be a mapping"
            )
        try:
            emit_rec = EmitRecommendation(
                action_name=str(r["action_name"]),
                tier=int(r["tier"]),
                reasoning=str(r["reasoning"]),
                confidence=float(r.get("confidence", 1.0)),
                args=r.get("args") if isinstance(r.get("args"), dict) else {},
                expected_duration_s=(
                    float(r["expected_duration_s"])
                    if "expected_duration_s" in r else None
                ),
            )
        except KeyError as e:
            raise TreeValidationError(
                f"tree {tree_id!r} node {node_id!r}: "
                f"emit_recommendation missing required field {e}"
            ) from e
        if emit_rec.tier not in (1, 2, 3):
            raise TreeValidationError(
                f"tree {tree_id!r} node {node_id!r}: "
                f"recommendation tier must be 1|2|3; got {emit_rec.tier}"
            )

    return Then(
        emit_thought=raw.get("emit_thought") if isinstance(raw.get("emit_thought"), str) else None,
        emit_verdict=emit_verdict,
        emit_recommendation=emit_rec,
        next=raw.get("next") if isinstance(raw.get("next"), str) else None,
        goto_tree=raw.get("goto_tree") if isinstance(raw.get("goto_tree"), str) else None,
        include_tree=raw.get("include_tree") if isinstance(raw.get("include_tree"), str) else None,
        stop=bool(raw.get("stop", False)),
    )


def _require(d: dict, key: str, expected_type: type) -> Any:
    if key not in d:
        raise TreeValidationError(f"missing required key {key!r}")
    v = d[key]
    if not isinstance(v, expected_type):
        raise TreeValidationError(
            f"key {key!r} must be {expected_type.__name__}; got {type(v).__name__}"
        )
    return v


# ---------------------------------------------------------------------------
# Registry — load all trees from a directory and cross-validate goto_tree refs
# ---------------------------------------------------------------------------


def load_tree_registry(
    trees_dir: Union[str, Path],
    known_diag_tools: set[str],
    known_action_names: set[str],
) -> dict[str, Tree]:
    """Load every *.yaml under trees_dir into a registry keyed by tree id.
    Validates that:
      - every node.diag is in known_diag_tools (or None for compute-only)
      - every emit_recommendation.action_name is in known_action_names
      - every goto_tree references a tree present in the registry
    """
    p = Path(trees_dir)
    if not p.is_dir():
        raise TreeValidationError(f"trees_dir {p} is not a directory")

    registry: dict[str, Tree] = {}
    for yaml_path in sorted(p.glob("*.yaml")):
        tree = load_tree_yaml(yaml_path)
        if tree.id in registry:
            raise TreeValidationError(
                f"duplicate tree id {tree.id!r} (also in {yaml_path})"
            )
        registry[tree.id] = tree

    if not registry:
        raise TreeValidationError(f"no *.yaml trees found under {p}")

    # Cross-validate.
    for tree in registry.values():
        _validate_tree_refs(tree, registry, known_diag_tools, known_action_names)
    return registry


def _validate_tree_refs(
    tree: Tree,
    registry: dict[str, Tree],
    known_diag_tools: set[str],
    known_action_names: set[str],
) -> None:
    for node in tree.nodes.values():
        if node.diag is not None and node.diag not in known_diag_tools:
            raise TreeValidationError(
                f"tree {tree.id!r} node {node.id!r}: "
                f"references unknown diag tool {node.diag!r}. "
                f"Known: {sorted(known_diag_tools)}"
            )
        for branch in node.branches:
            then = branch.then
            if then.emit_recommendation is not None:
                action = then.emit_recommendation.action_name
                if action not in known_action_names:
                    raise TreeValidationError(
                        f"tree {tree.id!r} node {node.id!r}: "
                        f"references unknown action {action!r}. "
                        f"Add to action_whitelist.json or fix tree."
                    )
            if then.goto_tree is not None and then.goto_tree not in registry:
                raise TreeValidationError(
                    f"tree {tree.id!r} node {node.id!r}: "
                    f"goto_tree references unknown tree {then.goto_tree!r}"
                )
            if then.include_tree is not None and then.include_tree not in registry:
                raise TreeValidationError(
                    f"tree {tree.id!r} node {node.id!r}: "
                    f"include_tree references unknown tree {then.include_tree!r}"
                )
    # Recursively validate sub-trees.
    for sub in tree.subtrees.values():
        _validate_tree_refs(sub, registry, known_diag_tools, known_action_names)
