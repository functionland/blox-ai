"""Tests for src/runtime/tree_expr.py — simpleeval wrapper + DotDict."""
from __future__ import annotations

import pytest

from src.runtime.tree_expr import (
    DotDict,
    ExpressionEvalError,
    ExpressionParseError,
    evaluate,
    to_dotdict,
    validate_expression,
)


# ---------------------------------------------------------------------------
# DotDict + to_dotdict
# ---------------------------------------------------------------------------


class TestDotDict:
    def test_attr_access_basic(self):
        d = DotDict({"a": 1, "b": "two"})
        assert d.a == 1
        assert d.b == "two"

    def test_attr_access_missing_raises_attributeerror(self):
        d = DotDict({"a": 1})
        with pytest.raises(AttributeError):
            d.b

    def test_nested_dict_auto_wraps(self):
        d = DotDict({"outer": {"inner": 42}})
        assert d.outer.inner == 42

    def test_nested_list_of_dicts_auto_wraps(self):
        # Note: avoid keys named like dict methods ("items", "keys",
        # "values", "get") — Python's attribute lookup hits the method
        # first, so `d.items` returns the bound method, not the field.
        # Use unambiguous names in trees.
        d = DotDict({"entries": [{"x": 1}, {"x": 2}]})
        assert d.entries[0].x == 1
        assert d.entries[1].x == 2

    def test_dict_methods_still_work(self):
        d = DotDict({"a": 1})
        assert d.get("a") == 1
        assert "a" in d


class TestToDotDict:
    def test_wraps_plain_dict(self):
        d = to_dotdict({"x": 1})
        assert isinstance(d, DotDict)

    def test_wraps_nested_dict(self):
        d = to_dotdict({"a": {"b": {"c": 5}}})
        assert d.a.b.c == 5

    def test_passes_through_primitives(self):
        assert to_dotdict(5) == 5
        assert to_dotdict("x") == "x"
        assert to_dotdict(None) is None

    def test_wraps_list_of_dicts(self):
        out = to_dotdict([{"a": 1}, {"a": 2}, 3])
        assert isinstance(out[0], DotDict)
        assert out[0].a == 1
        assert out[2] == 3


# ---------------------------------------------------------------------------
# validate_expression — parse-only
# ---------------------------------------------------------------------------


class TestValidateExpression:
    def test_accepts_simple_comparison(self):
        validate_expression("result.x == 1")

    def test_accepts_boolean_combinations(self):
        validate_expression("result.x == 1 and result.y > 5 or not result.z")

    def test_accepts_compound_types(self):
        validate_expression("len(result.entries) > 0")

    def test_rejects_syntax_error(self):
        with pytest.raises(ExpressionParseError):
            validate_expression("result.x ==")

    def test_rejects_empty(self):
        with pytest.raises(ExpressionParseError):
            validate_expression("")


# ---------------------------------------------------------------------------
# evaluate — happy path
# ---------------------------------------------------------------------------


class TestEvaluateHappy:
    def test_simple_comparison(self):
        ctx = {"result": to_dotdict({"x": 5})}
        assert evaluate("result.x == 5", ctx) is True
        assert evaluate("result.x == 6", ctx) is False

    def test_boolean_and_or_not(self):
        ctx = {"result": to_dotdict({"a": True, "b": False})}
        assert evaluate("result.a and not result.b", ctx) is True

    def test_nested_field_access(self):
        ctx = {"result": to_dotdict({"deep": {"inner": "match"}})}
        assert evaluate("result.deep.inner == 'match'", ctx) is True

    def test_field_via_list_index(self):
        ctx = {"result": to_dotdict({"entries": [{"name": "a"}, {"name": "b"}]})}
        assert evaluate("result.entries[1].name == 'b'", ctx) is True

    def test_safe_helper_isnull(self):
        ctx = {"result": to_dotdict({"opt": None})}
        assert evaluate("isnull(result.opt)", ctx) is True
        ctx2 = {"result": to_dotdict({"opt": 5})}
        assert evaluate("isnull(result.opt)", ctx2) is False

    def test_safe_helper_len_on_list(self):
        ctx = {"result": to_dotdict({"entries": [1, 2, 3]})}
        assert evaluate("len(result.entries) == 3", ctx) is True

    def test_safe_helper_any_with_generator(self):
        ctx = {"result": to_dotdict({"entries": [
            {"up_to_date": True},
            {"up_to_date": False},
            {"up_to_date": True},
        ]})}
        assert evaluate(
            "any(item.up_to_date == False for item in result.entries)",
            ctx,
        ) is True

    def test_yaml_friendly_lowercase_true_false_alias(self):
        """Tree YAML authors often write `true`/`false` (lowercase).
        Our context aliases match."""
        ctx = {"result": to_dotdict({"ok": True}), "true": True, "false": False}
        assert evaluate("result.ok == true", ctx) is True


# ---------------------------------------------------------------------------
# evaluate — guard rails
# ---------------------------------------------------------------------------


class TestEvaluateGuards:
    def test_undefined_name_raises_eval_error(self):
        ctx = {"result": to_dotdict({"x": 1})}
        with pytest.raises(ExpressionEvalError):
            evaluate("nonexistent_name == 1", ctx)

    def test_undefined_field_raises_eval_error(self):
        ctx = {"result": to_dotdict({"x": 1})}
        with pytest.raises(ExpressionEvalError):
            evaluate("result.missing_field == 1", ctx)

    def test_blocks_dunder_attribute_access(self):
        """The classic sandbox escape — accessing __class__ / __mro__ /
        __subclasses__ to walk the object graph. simpleeval blocks this
        by default; DotDict must not re-open the door."""
        ctx = {"result": to_dotdict({"x": 1})}
        with pytest.raises(ExpressionEvalError):
            evaluate("result.__class__", ctx)

    def test_blocks_function_calls_not_on_whitelist(self):
        """`exec`, `eval`, `open`, `__import__` etc. must not be callable."""
        ctx = {"result": to_dotdict({})}
        with pytest.raises(ExpressionEvalError):
            evaluate("__import__('os').system('whoami')", ctx)
        with pytest.raises(ExpressionEvalError):
            evaluate("open('/etc/passwd').read()", ctx)
        with pytest.raises(ExpressionEvalError):
            evaluate("eval('1+1')", ctx)

    def test_power_max_blocks_resource_exhaustion(self):
        """simpleeval's POWER_MAX default (4_000_000) rejects huge
        exponents before evaluation. `10 ** 5_000_000` would overflow
        memory if it executed."""
        ctx = {}
        with pytest.raises(ExpressionEvalError):
            evaluate("10 ** 5000000", ctx)
