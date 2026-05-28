"""Tree runner — walks a Tree per the DSL, calling diag/* tools and
yielding the SAME SSE event vocabulary the AI bridge yields.

Compatible with the existing session/event_buffer/detached-task
infrastructure (committed in 1a420fd). Apps/box's resume protocol
works for tree runs unchanged.

Walking semantics:
  1. Start at tree.entry (or the resolved subtree's entry).
  2. Optionally call the node's `diag` tool; await result.
  3. Cache the result in TreeRunContext (per-tree-run memoization;
     repeat-diag inside the same run hits the cache).
  4. Walk branches IN ORDER; the first one whose `when:` evaluates to
     truthy (or `default:`) wins.
  5. Execute the `then:` block:
        - emit thought / verdict / recommendation events
        - if goto_tree: switch to the named tree and continue
        - if stop: halt
        - else: jump to `then.next` node
  6. If no branch matched (no default and nothing truthy), halt.
  7. After the walk completes (stop or fall-off), if no verdict was
     emitted, synthesize one from the visited path.

Cycle detection: a global cap (`MAX_NODES_PER_RUN`) bounds the walk.
Trees should be acyclic but a malformed tree referencing itself
mustn't hang the session forever.

Cross-tree redirect: `goto_tree` re-enters at that tree's entry node.
The destination tree's id is pushed onto a stack so we can detect
loops (a → goes-to → b → goes-to → a) within MAX_TREE_HOPS.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from src.runtime.tree_dsl import Branch, Node, Then, Tree
from src.runtime.tree_expr import (
    ExpressionEvalError,
    evaluate,
    to_dotdict,
)


logger = logging.getLogger("blox-ai.tree-runner")


# Hard caps — defensive limits, not policy.
MAX_NODES_PER_RUN = 100      # any tree visiting >100 nodes is a bug
MAX_TREE_HOPS = 10           # goto_tree depth cap (cross-tree loops)


# Type alias for the diag executor we call.
# Mirrors RealDiagExecutor.__call__ signature so the tree runner can
# be wired with either RealDiagExecutor (production) or MockDiagExecutor
# (tests + C2).
DiagExecutor = Callable[[str, dict], Awaitable[dict]]


@dataclass
class TreeRunContext:
    """Per-run state — diag results cache + facts accumulator + visited
    node ids (for cycle bound + final summary).

    `red_verdict_emitted`: only RED verdicts (definitive problems)
    halt include_tree callers. Yellow + green verdicts let the caller
    continue — so when not-earning INCLUDES disconnected and
    disconnected emits a yellow "everything looks healthy" verdict,
    not-earning continues to its earnings-specific checks."""
    facts: dict = field(default_factory=dict)
    diag_cache: dict = field(default_factory=dict)   # (tool, args_key) → result
    visited: list[tuple[str, str]] = field(default_factory=list)
    verdict_emitted: bool = False
    red_verdict_emitted: bool = False
    recommendations_emitted: int = 0
    tree_stack: list[str] = field(default_factory=list)


class TreeRunner:
    """Async iterator over SSE-compatible events for a single tree run.

    Usage:
        runner = TreeRunner(trees=registry, tool_executor=executor)
        async for event in runner.run("disconnected"):
            await session.append_event(event)
    """

    def __init__(
        self,
        trees: dict[str, Tree],
        tool_executor: DiagExecutor,
    ):
        self.trees = trees
        self.tool_executor = tool_executor

    async def run(self, scenario_id: str) -> AsyncIterator[dict]:
        """Walk the tree for `scenario_id`. Yields SSE events. Caller
        is responsible for buffering / emitting them to the wire."""
        if scenario_id not in self.trees:
            yield self._error_event(
                "unknown_scenario",
                f"no tree registered for scenario_id={scenario_id!r}",
                recoverable=False,
            )
            return

        ctx = TreeRunContext()
        try:
            async for ev in self._walk_tree(self.trees[scenario_id], ctx):
                yield ev
        except Exception as e:
            logger.exception("tree-runner crashed for scenario=%s", scenario_id)
            yield self._error_event(
                "tree_runner_internal_error",
                f"{type(e).__name__}: {e}"[:1500],
                recoverable=True,
            )
            return

        # Fall-off path: no verdict was emitted explicitly. Synthesize
        # a 'no_definitive_finding' verdict so the app always has SOME
        # final state to render (matches the AI bridge's post-loop
        # verdict synthesis pattern).
        if not ctx.verdict_emitted:
            yield {
                "type": "verdict",
                "payload": {
                    "summary": "Tree completed without a definitive finding.",
                    "severity": "yellow",
                    "root_cause": "tree_indeterminate",
                },
            }

    # ----- internals -------------------------------------------------------

    async def _walk_tree(
        self, tree: Tree, ctx: TreeRunContext,
    ) -> AsyncIterator[dict]:
        """Walk a tree starting at its entry node. Handles `next:` jumps
        + `goto_tree:` redirects (with hop cap)."""
        if len(ctx.tree_stack) >= MAX_TREE_HOPS:
            yield self._error_event(
                "tree_hop_limit",
                f"max tree-redirect depth ({MAX_TREE_HOPS}) reached; "
                f"stack: {ctx.tree_stack}",
                recoverable=True,
            )
            return
        ctx.tree_stack.append(tree.id)
        try:
            async for ev in self._walk_from(tree, tree.entry, ctx):
                yield ev
        finally:
            ctx.tree_stack.pop()

    async def _walk_from(
        self, tree: Tree, start_node_id: str, ctx: TreeRunContext,
    ) -> AsyncIterator[dict]:
        """Walk nodes starting at start_node_id. Stops on `stop: true`,
        unmatched branch (no default), or fall-off the cascade."""
        current_id: Optional[str] = start_node_id
        steps_in_this_tree = 0

        while current_id is not None:
            if len(ctx.visited) >= MAX_NODES_PER_RUN:
                yield self._error_event(
                    "tree_node_limit",
                    f"max nodes per run ({MAX_NODES_PER_RUN}) reached "
                    f"(possible cycle in tree {tree.id!r})",
                    recoverable=True,
                )
                return
            steps_in_this_tree += 1

            # Sub-tree dispatch — `next:` may reference a subtree id.
            if current_id in tree.subtrees:
                async for ev in self._walk_tree(tree.subtrees[current_id], ctx):
                    yield ev
                return  # sub-tree ended the walk

            if current_id not in tree.nodes:
                yield self._error_event(
                    "missing_node",
                    f"tree {tree.id!r}: node {current_id!r} not found",
                    recoverable=True,
                )
                return

            node = tree.nodes[current_id]
            ctx.visited.append((tree.id, node.id))

            # 1. Optionally run the node's diag tool.
            diag_result = None
            if node.diag is not None:
                tool_name = f"diag/{node.diag}"
                async for ev in self._run_diag(
                    tool_name, node.diag_args, node.timeout_s, ctx,
                ):
                    yield ev
                cache_key = self._cache_key(tool_name, node.diag_args)
                diag_result = ctx.diag_cache.get(cache_key)

            # 2. Pick the matching branch.
            matched, branch = self._match_branch(node, diag_result, ctx)
            if not matched:
                # No default + no condition matched → halt this cascade.
                # Don't error; trees may intentionally end here.
                return

            then = branch.then

            # 3. Execute the then-block. emit ordering: thought first
            # (so the user sees context before the verdict), then the
            # verdict, then the recommendation.
            if then.emit_thought:
                yield {"type": "thought", "payload": then.emit_thought[:4000]}

            if then.emit_verdict is not None:
                yield {
                    "type": "verdict",
                    "payload": {
                        "summary":    then.emit_verdict.summary[:500],
                        "severity":   then.emit_verdict.severity,
                        "root_cause": then.emit_verdict.root_cause[:200],
                    },
                }
                ctx.verdict_emitted = True
                if then.emit_verdict.severity == "red":
                    ctx.red_verdict_emitted = True

            if then.emit_recommendation is not None:
                yield self._make_recommendation_event(then.emit_recommendation)
                ctx.recommendations_emitted += 1

            # 4. Decide next step.
            if then.stop:
                return
            if then.goto_tree is not None:
                # Cross-tree redirect: walk the target tree and end the
                # current cascade. (We don't return to this node after
                # the redirect completes; trees are intended to express
                # full handoff with goto_tree.)
                if then.goto_tree not in self.trees:
                    yield self._error_event(
                        "unknown_goto_tree",
                        f"tree {tree.id!r} node {node.id!r}: "
                        f"goto_tree={then.goto_tree!r} not registered",
                        recoverable=True,
                    )
                    return
                async for ev in self._walk_tree(self.trees[then.goto_tree], ctx):
                    yield ev
                return
            if then.include_tree is not None:
                # Cross-tree CALL: walk the target tree, then return.
                # If the called tree emitted a verdict, the cascade
                # halts (we have a definitive answer; no point
                # continuing the caller). Otherwise continue with
                # `next:`. The OUTER run() does the synth-verdict
                # post-walk; _walk_tree itself never synths, so
                # `verdict_emitted` flipping during the include is
                # always a real explicit verdict.
                if then.include_tree not in self.trees:
                    yield self._error_event(
                        "unknown_include_tree",
                        f"tree {tree.id!r} node {node.id!r}: "
                        f"include_tree={then.include_tree!r} not registered",
                        recoverable=True,
                    )
                    return
                red_before = ctx.red_verdict_emitted
                async for ev in self._walk_tree(self.trees[then.include_tree], ctx):
                    yield ev
                # Halt caller only on RED verdicts (definitive problems).
                # Yellow/green verdicts from the included tree let the
                # caller continue with its own scenario-specific checks —
                # so when not-earning INCLUDES disconnected and
                # disconnected emits a yellow "everything looks healthy"
                # verdict, not-earning continues checking cluster + chain.
                if ctx.red_verdict_emitted and not red_before:
                    return   # included tree found a definitive answer
                # else: fall through to `current_id = then.next` below
            current_id = then.next   # may be None → halt

    def _match_branch(
        self,
        node: Node,
        diag_result: Optional[dict],
        ctx: TreeRunContext,
    ) -> tuple[bool, Optional[Branch]]:
        """Walk branches in order; return (True, first-matching-branch)
        or (False, None) if nothing matched (including no default)."""
        eval_ctx = {
            "result": to_dotdict(diag_result if diag_result is not None else {}),
            "facts":  to_dotdict(ctx.facts),
            "True":   True, "False": False, "None": None,
            "true":   True, "false": False, "null": None,
        }
        default_branch: Optional[Branch] = None
        for branch in node.branches:
            if branch.when is None:
                default_branch = branch
                continue
            try:
                if evaluate(branch.when, eval_ctx):
                    return True, branch
            except ExpressionEvalError as e:
                # An expression that raises is treated as False for THIS
                # branch (defensive — a malformed result.foo lookup
                # shouldn't kill the whole tree run). Log so operators
                # see it; trees with chronic eval errors should be
                # caught by Phase 1.a load-time `validate_expression`.
                logger.warning(
                    "tree %s node %s branch when=%r: eval error %s — treating as False",
                    ctx.tree_stack[-1] if ctx.tree_stack else "?",
                    node.id, branch.when, e,
                )
        if default_branch is not None:
            return True, default_branch
        return False, None

    async def _run_diag(
        self,
        tool: str,
        args: dict,
        timeout_s: float,
        ctx: TreeRunContext,
    ) -> AsyncIterator[dict]:
        """Run a diag tool with per-tree-run memoization. Emits the
        tool_call + tool_result events the AI bridge would emit, so the
        app's transcript renderer treats tree runs identically to AI
        runs."""
        cache_key = self._cache_key(tool, args)
        if cache_key in ctx.diag_cache:
            # No event emit on cache hit — apps would otherwise see
            # phantom "called diag/X" events for cheap re-checks.
            return

        call_id = secrets.token_hex(8)
        yield {
            "type": "tool_call",
            "call_id": call_id,
            "payload": {"tool": tool, "args": args},
        }
        try:
            result = await asyncio.wait_for(
                self.tool_executor(tool, args), timeout=timeout_s,
            )
            ctx.diag_cache[cache_key] = result if isinstance(result, dict) else {}
            yield {
                "type": "tool_result",
                "call_id": call_id,
                "ok": True,
                "payload": result if isinstance(result, dict) else {"value": result},
            }
        except asyncio.TimeoutError:
            ctx.diag_cache[cache_key] = {}
            yield {
                "type": "tool_result",
                "call_id": call_id,
                "ok": False,
                "error": f"{tool} timed out after {timeout_s}s",
                "payload": {},
            }
        except Exception as e:
            ctx.diag_cache[cache_key] = {}
            yield {
                "type": "tool_result",
                "call_id": call_id,
                "ok": False,
                "error": f"{type(e).__name__}: {e}"[:2000],
                "payload": {},
            }

    @staticmethod
    def _cache_key(tool: str, args: dict) -> tuple:
        """Stable hashable key for diag memoization."""
        return (tool, tuple(sorted(args.items())))

    @staticmethod
    def _make_recommendation_event(rec) -> dict:
        """Build a recommended_action event matching sse_events.schema.json.
        The approval_token is a 64-byte cryptographic random — the
        executor will validate it against per-session secret HMAC."""
        action_id = secrets.token_hex(8)
        # 64-byte hex = 128 chars; schema requires minLength=64 maxLength=2048.
        approval_token = secrets.token_hex(32)   # 64 hex chars
        event: dict = {
            "type": "recommended_action",
            "action_id": action_id,
            "action_name": rec.action_name,
            "args": rec.args,
            "reasoning": rec.reasoning[:1000],
            "confidence": rec.confidence,
            "tier": rec.tier,
            "approval_token": approval_token,
        }
        if rec.expected_duration_s is not None:
            event["expected_duration_s"] = rec.expected_duration_s
        return event

    @staticmethod
    def _error_event(code: str, message: str, *, recoverable: bool) -> dict:
        return {
            "type": "error",
            "code": code[:64],
            "message": message[:2000],
            "recoverable": recoverable,
        }
