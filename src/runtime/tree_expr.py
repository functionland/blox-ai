"""Safe expression evaluator for tree branch `when:` conditions.

Wraps simpleeval with a whitelist-only contract:
  - Operators: ==, !=, <, >, <=, >=, and, or, not, in, +, -, *, /
    (Python boolean operators; YAML authors write Python syntax)
  - Names: only the symbols passed in via the `context` dict for each
    evaluation (typically `result`, `facts`, plus literal True/False/None)
  - No function calls EXCEPT the small whitelist below (length, any,
    all, isnull, get) — needed for tree expressions like
    `any(c.up_to_date == false for c in result.values())`
  - No attribute access on imports, no __builtins__, no exec/eval

The two known attack surfaces against a YAML-loaded expression
evaluator are:
  1. Calls into Python internals (__class__, __mro__, __subclasses__).
     simpleeval blocks attribute access on objects by default, but
     we EXTEND access to walk our result dicts via DotDict.__getattr__
     — so the DotDict implementation must NOT expose any methods that
     leak the dict's containing object.
  2. Resource exhaustion via huge literals (10**1000). simpleeval has
     POWER_MAX, MAX_STRING_LENGTH built-in; we keep the defaults.

If a tree expression syntactically can't compile, validate_expression
raises ExpressionParseError at LOAD time, NOT at run time. Trees with
broken expressions fail fast on container start.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from simpleeval import (
    EvalWithCompoundTypes,
    InvalidExpression,
    SimpleEval,
    DEFAULT_FUNCTIONS,
)


logger = logging.getLogger("blox-ai.tree-expr")


class ExpressionParseError(ValueError):
    """A tree's `when:` expression failed to compile."""


class ExpressionEvalError(RuntimeError):
    """A tree's `when:` expression compiled but failed at run time
    (KeyError on an undefined result field, ZeroDivisionError, etc.).
    Tree runner catches + treats as `False` for the branch."""


# Whitelist of safe builtins the YAML can reference inside `when:`.
# Kept VERY small — every entry is a tested attack surface.
_SAFE_FUNCTIONS = {
    "len":   len,
    "any":   any,
    "all":   all,
    "min":   min,
    "max":   max,
    # Custom helper — null/None check (YAML authors are more comfortable
    # writing `isnull(x)` than `x is None`).
    "isnull": lambda x: x is None,
    # Safe dict.get equivalent — for fields that might be absent without
    # bombing the whole expression. Usage: `get(result, 'optional_field', 0) > 5`.
    "get":   lambda d, k, default=None: (d.get(k, default) if isinstance(d, dict) else getattr(d, k, default)),
}


class DotDict(dict):
    """dict subclass that allows attribute access (.key) for nested
    field reads inside tree expressions. Nested dicts are wrapped
    lazily on attribute access so the original dict isn't mutated.

    Security-relevant: __getattr__ only delegates to dict.__getitem__.
    No method calls leak. If a result field happens to be named the
    same as a dict method (e.g. 'get'), the dict method wins (Python
    attribute lookup order) — that's a tree-author concern, not a
    security one."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError:
            raise AttributeError(name) from None
        if isinstance(v, dict) and not isinstance(v, DotDict):
            v = DotDict(v)
            # Cache the wrapped form in-place so repeated access is cheap.
            self[name] = v
        elif isinstance(v, list):
            v = [DotDict(item) if isinstance(item, dict) and not isinstance(item, DotDict) else item
                 for item in v]
            self[name] = v
        return v


def to_dotdict(value: Any) -> Any:
    """Recursively wrap dicts in DotDict so tree expressions can use
    dot-attribute syntax. Lists are walked; primitives passed through."""
    if isinstance(value, dict):
        out = DotDict({k: to_dotdict(v) for k, v in value.items()})
        return out
    if isinstance(value, list):
        return [to_dotdict(v) for v in value]
    return value


def _make_evaluator(names: Mapping[str, Any]) -> EvalWithCompoundTypes:
    """Build a per-evaluation simpleeval instance. CompoundTypes lets
    expressions reference dicts + lists; the default SimpleEval does not."""
    evl = EvalWithCompoundTypes(
        names=dict(names),
        functions={**DEFAULT_FUNCTIONS, **_SAFE_FUNCTIONS},
    )
    return evl


def _normalize(expr: str) -> str:
    """Collapse YAML `|` literal-block newlines into spaces so authors
    can split long expressions across multiple lines for readability.
    simpleeval delegates to ast.parse which rejects bare newlines
    inside binary ops (`a == 1 and\nb == 2` is a SyntaxError)."""
    return " ".join(expr.split())


def validate_expression(expr: str) -> None:
    """Parse-check an expression without evaluating. Called at tree
    LOAD time so a broken `when:` syntax fails before any troubleshoot
    session starts. Raises ExpressionParseError on failure."""
    norm = _normalize(expr)
    if not norm:
        raise ExpressionParseError("empty expression")
    try:
        evl = _make_evaluator({})
        evl.parse(norm)
    except (SyntaxError, InvalidExpression) as e:
        raise ExpressionParseError(f"cannot parse {expr!r}: {e}") from e
    except Exception as e:
        raise ExpressionParseError(f"cannot parse {expr!r}: {e}") from e


def evaluate(expr: str, context: Mapping[str, Any]) -> bool:
    """Evaluate `expr` against `context`. Returns the boolean result.
    Raises ExpressionEvalError if the expression references undefined
    names OR raises at run time (KeyError / TypeError / etc).

    The caller is expected to have wrapped any dict-shaped values in
    DotDict (via to_dotdict) so attribute access works."""
    norm = _normalize(expr)
    try:
        evl = _make_evaluator(context)
        result = evl.eval(norm)
    except InvalidExpression as e:
        raise ExpressionEvalError(f"cannot evaluate {expr!r}: {e}") from e
    except (KeyError, AttributeError, TypeError, ZeroDivisionError) as e:
        raise ExpressionEvalError(f"runtime error in {expr!r}: {e}") from e
    return bool(result)
