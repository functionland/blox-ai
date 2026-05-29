"""C4 — ActionExecutor unit tests.

Covers:
  - whitelist load + hash determinism
  - action_not_in_whitelist rejection
  - args_constraint_violation rejection
  - approval_token gates (invalid / expired / replayed / mismatched id)
  - tier-3 security_code gates (missing file / missing field / wrong)
  - tier-2 happy path → executed=true audit line
  - tier-3 happy path → executed=true audit line (security_code_valid=true)
  - flag-file dispatch path (maps_to_core=true)
  - subprocess dispatch path (docker.restart with arg constraint)
  - audit line conditional invariants (executed=true requires result;
    executed=false requires non-empty rejected_reason + no result)
  - whitelist_hash stamped on every line + identical across requests
    when the file hasn't changed
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.approval_token import ApprovalTokenSigner
from src.tools.executor import (
    ActionExecutor,
    WhitelistError,
    load_whitelist,
    read_security_code,
)


def _real_whitelist_or_skip():
    from .conftest import _locate_fula_ota_api_dir
    api = _locate_fula_ota_api_dir()
    if api is None:
        pytest.skip("requires fula-ota sibling checkout / env var")
    wl = api.parent / "action_whitelist.json"
    if not wl.is_file():
        pytest.skip("action_whitelist.json not present alongside schemas")
    return wl


@pytest.fixture
def executor(tmp_path, monkeypatch):
    wl_src = _real_whitelist_or_skip()
    wl_path = tmp_path / "action_whitelist.json"
    wl_path.write_bytes(wl_src.read_bytes())
    monkeypatch.setenv("BLOX_AI_APPROVAL_SECRET_PATH",
                       str(tmp_path / "approval-secret"))
    sec_path = tmp_path / "security-code"
    sec_path.write_text("1234")
    monkeypatch.setenv("BLOX_AI_SECURITY_CODE_PATH", str(sec_path))
    flag_dir = tmp_path / "commands"
    flag_dir.mkdir()
    monkeypatch.setenv("BLOX_AI_COMMANDS_FLAG_DIR", str(flag_dir))
    signer = ApprovalTokenSigner()
    wl = load_whitelist(str(wl_path))
    audit = tmp_path / "audit.jsonl"
    ex = ActionExecutor(signer=signer, whitelist=wl, audit_path=str(audit))
    ex._audit_path_for_test = str(audit)  # convenience for tests
    ex._tmp_path = tmp_path
    return ex


def _last_audit_line(executor) -> dict:
    lines = Path(executor._audit_path_for_test).read_text().splitlines()
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Whitelist loader
# ---------------------------------------------------------------------------

def test_whitelist_load_real_file(tmp_path):
    wl_src = _real_whitelist_or_skip()
    wl_path = tmp_path / "action_whitelist.json"
    wl_path.write_bytes(wl_src.read_bytes())
    wl = load_whitelist(str(wl_path))
    assert "docker.restart" in wl.tier_2_names
    assert "reset" in wl.tier_3_names
    assert "restart_fula" in wl.tier_2_maps_to_core
    assert "docker.restart" not in wl.tier_2_maps_to_core  # subprocess
    assert wl.arg_constraints["docker.restart"]["container"]


def test_whitelist_hash_deterministic(tmp_path):
    wl_src = _real_whitelist_or_skip()
    wl_path = tmp_path / "action_whitelist.json"
    wl_path.write_bytes(wl_src.read_bytes())
    a = load_whitelist(str(wl_path))
    b = load_whitelist(str(wl_path))
    assert a.sha256_hex == b.sha256_hex
    assert len(a.sha256_hex) == 64


def test_whitelist_load_raises_on_missing(tmp_path):
    with pytest.raises(WhitelistError):
        load_whitelist(str(tmp_path / "no-such-file.json"))


def test_whitelist_load_raises_on_bad_json(tmp_path):
    p = tmp_path / "wl.json"
    p.write_text("{not json")
    with pytest.raises(WhitelistError):
        load_whitelist(str(p))


# ---------------------------------------------------------------------------
# Security code
# ---------------------------------------------------------------------------

def test_read_security_code_present(tmp_path, monkeypatch):
    p = tmp_path / "sc"
    p.write_text("5678")
    monkeypatch.setenv("BLOX_AI_SECURITY_CODE_PATH", str(p))
    from src.tools.executor import read_security_code
    assert read_security_code() == "5678"


def test_read_security_code_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOX_AI_SECURITY_CODE_PATH",
                       str(tmp_path / "nope"))
    from src.tools.executor import read_security_code
    assert read_security_code() is None


def test_read_security_code_empty(tmp_path, monkeypatch):
    p = tmp_path / "sc"
    p.write_text("   \n")
    monkeypatch.setenv("BLOX_AI_SECURITY_CODE_PATH", str(p))
    from src.tools.executor import read_security_code
    assert read_security_code() is None


# ---------------------------------------------------------------------------
# Executor rejections
# ---------------------------------------------------------------------------

def _exec_sync(executor, **kwargs):
    """Sync runner for async executor.execute via asyncio.run."""
    import asyncio
    return asyncio.run(executor.execute(**kwargs))


def test_reject_unknown_action(executor):
    token = executor.signer.sign("act-1")
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code=None,
        action_name="never-heard-of-this",
        action_args={},
    )
    assert r["http_status"] == 403
    line = _last_audit_line(executor)
    assert line["rejected_reason"] == "action_not_in_whitelist"
    assert line["executed"] is False
    assert "result" not in line  # conditional invariant


def test_reject_args_constraint_violation(executor):
    token = executor.signer.sign("act-1")
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code=None,
        action_name="docker.restart",
        action_args={"container": "evil-container"},
    )
    assert r["http_status"] == 403
    line = _last_audit_line(executor)
    assert line["rejected_reason"] == "args_constraint_violation"
    assert line["executed"] is False


def test_reject_token_invalid(executor):
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token="!!!" + "x" * 64,
        security_code=None,
        action_name="docker.restart",
        action_args={"container": "ipfs_cluster"},
    )
    assert r["http_status"] == 401
    line = _last_audit_line(executor)
    assert line["rejected_reason"] == "approval_token_invalid"


def test_reject_token_replayed(executor):
    token = executor.signer.sign("act-1")
    # First call consumes nonce + succeeds with subprocess mock
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(args=[], returncode=0,
                                            stdout="ok", stderr="")
        _exec_sync(
            executor,
            action_id="act-1",
            approval_token=token,
            security_code=None,
            action_name="docker.restart",
            action_args={"container": "ipfs_cluster"},
        )
    # Replay
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code=None,
        action_name="docker.restart",
        action_args={"container": "ipfs_cluster"},
    )
    assert r["http_status"] == 401
    line = _last_audit_line(executor)
    assert line["rejected_reason"] == "approval_token_replayed"


def test_reject_tier3_missing_security_code(executor):
    token = executor.signer.sign("act-1")
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code=None,
        action_name="reset",
        action_args={},
    )
    assert r["http_status"] == 403
    line = _last_audit_line(executor)
    assert line["rejected_reason"] == "security_code_required_but_missing"


def test_reject_tier3_wrong_security_code(executor):
    token = executor.signer.sign("act-1")
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code="WRONG",
        action_name="reset",
        action_args={},
    )
    assert r["http_status"] == 403
    line = _last_audit_line(executor)
    assert line["rejected_reason"] == "security_code_invalid"
    assert line["security_code_valid"] is False


def test_reject_tier3_security_code_file_missing(executor, tmp_path, monkeypatch):
    # Remove the security code file (override env var; per-call read picks it up)
    monkeypatch.setenv("BLOX_AI_SECURITY_CODE_PATH",
                       str(tmp_path / "no-such-sec-file"))
    token = executor.signer.sign("act-1")
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code="anything",
        action_name="reset",
        action_args={},
    )
    assert r["http_status"] == 403
    line = _last_audit_line(executor)
    assert line["rejected_reason"] == "security_code_file_missing"


# ---------------------------------------------------------------------------
# Executor happy paths
# ---------------------------------------------------------------------------

def test_tier2_subprocess_happy_path(executor):
    token = executor.signer.sign("act-1")
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(args=[], returncode=0,
                                            stdout="restarted", stderr="")
        r = _exec_sync(
            executor,
            action_id="act-1",
            approval_token=token,
            security_code=None,
            action_name="docker.restart",
            action_args={"container": "ipfs_cluster"},
        )
    assert r["http_status"] == 200
    line = _last_audit_line(executor)
    assert line["executed"] is True
    assert line["rejected_reason"] == ""  # conditional invariant
    assert "result" in line
    assert line["result"]["success"] is True
    assert line["result"]["exit_code"] == 0


def test_tier2_flag_file_dispatch(executor):
    """maps_to_core=true action — should touch /commands/.command_<name>."""
    token = executor.signer.sign("act-1")
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code=None,
        action_name="restart_fula",
        action_args={},
    )
    assert r["http_status"] == 200
    flag = executor._tmp_path / "commands" / ".command_restart_fula"
    assert flag.exists()


def test_tier3_happy_path_with_correct_security_code(executor):
    token = executor.signer.sign("act-1")
    r = _exec_sync(
        executor,
        action_id="act-1",
        approval_token=token,
        security_code="1234",
        action_name="reset",
        action_args={},
    )
    assert r["http_status"] == 200
    line = _last_audit_line(executor)
    assert line["executed"] is True
    assert line["security_code_valid"] is True
    assert line["tier"] == 3


# ---------------------------------------------------------------------------
# Audit line invariants
# ---------------------------------------------------------------------------

def test_whitelist_hash_stamped_on_every_line(executor):
    # 3 rejects, 1 execute — all should carry the same whitelist_hash
    for _ in range(3):
        _exec_sync(executor, action_id="x", approval_token="bad",
                   security_code=None, action_name="docker.restart",
                   action_args={"container": "ipfs_cluster"})
    token = executor.signer.sign("y")
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(args=[], returncode=0,
                                            stdout="", stderr="")
        _exec_sync(executor, action_id="y", approval_token=token,
                   security_code=None, action_name="docker.restart",
                   action_args={"container": "ipfs_cluster"})
    lines = [json.loads(L) for L in
             Path(executor._audit_path_for_test).read_text().splitlines()]
    assert len(lines) == 4
    hashes = {L["whitelist_hash"] for L in lines}
    assert len(hashes) == 1
    h = next(iter(hashes))
    assert len(h) == 64


def test_audit_line_required_fields_present(executor):
    """All schema-required fields appear on every line."""
    _exec_sync(executor, action_id="x", approval_token="bad",
               security_code=None, action_name="docker.restart",
               action_args={"container": "ipfs_cluster"})
    line = _last_audit_line(executor)
    required = {"ts", "request_id", "action_id", "action", "args", "tier",
                "approval_token_valid", "security_code_required", "executed",
                "approver_transport", "duration_ms", "executor_version",
                "whitelist_hash"}
    assert required.issubset(line.keys())


# ---------------------------------------------------------------------------
# HTTP status mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason,expected_status", [
    ("approval_token_invalid", 401),
    ("approval_token_expired", 401),
    ("approval_token_replayed", 401),
    ("action_not_in_whitelist", 403),
    ("args_constraint_violation", 403),
    ("security_code_invalid", 403),
    ("security_code_file_missing", 403),
    ("security_code_required_but_missing", 403),
    ("executor_busy", 429),
    ("recommendation_not_found", 409),
    ("internal_error", 500),
])
def test_status_map_for_each_rejection(executor, reason, expected_status):
    """Quick sanity that the reason→status map is right."""
    line = {
        "request_id": "r", "action_id": "x", "action": "y",
        "args": {}, "tier": 0, "approval_token_valid": False,
        "security_code_required": False, "approver_transport": "ble",
        "duration_ms": 0, "executor_version": "0.1.0",
        "whitelist_hash": "h", "ts": "t", "executed": False,
        "rejected_reason": "",
    }
    out = executor._reject(line, reason, time.monotonic())
    assert out["http_status"] == expected_status


# ---------------------------------------------------------------------------
# Additional coverage gaps surfaced by the post-impl advisor review
# ---------------------------------------------------------------------------

def test_concurrent_execute_hits_executor_busy(executor):
    """Two concurrent execute() calls — the second should reject with
    executor_busy. Exercises the actual asyncio.Lock concurrency path
    (not just the synthesized _reject helper)."""
    import asyncio

    async def runner():
        token_a = executor.signer.sign("a")
        token_b = executor.signer.sign("b")
        with patch("src.tools.executor.subprocess.run") as sub:
            from subprocess import CompletedProcess
            # Make subprocess slow enough that the second call sees the
            # lock held. We use a closure with asyncio.sleep via the
            # run_in_executor return-value mechanism — but subprocess.run
            # is sync. Easier: use a real long-ish sleep with time.sleep.
            import time as _time
            def slow_run(*a, **kw):
                _time.sleep(0.5)
                return CompletedProcess(args=[], returncode=0,
                                        stdout="ok", stderr="")
            sub.side_effect = slow_run
            results = await asyncio.gather(
                executor.execute(
                    action_id="a", approval_token=token_a,
                    security_code=None,
                    action_name="docker.restart",
                    action_args={"container": "ipfs_cluster"},
                ),
                executor.execute(
                    action_id="b", approval_token=token_b,
                    security_code=None,
                    action_name="docker.restart",
                    action_args={"container": "ipfs_cluster"},
                ),
            )
        return results
    results = asyncio.run(runner())
    statuses = sorted(r["http_status"] for r in results)
    assert statuses == [200, 429], (
        f"expected one 200 + one 429, got {statuses}"
    )
    busy_result = next(r for r in results if r["http_status"] == 429)
    assert busy_result["audit_line"]["rejected_reason"] == "executor_busy"


def test_subprocess_failure_records_success_false(executor):
    """When the subprocess runs but returns non-zero, executed=true +
    result.success=false. This distinguishes 'we couldn't run it' from
    'we ran it and it failed' — load-bearing for the conditional
    invariants on the audit line."""
    token = executor.signer.sign("act-fail")
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(
            args=[], returncode=1,
            stdout="", stderr="No such container: ipfs_cluster",
        )
        r = _exec_sync(
            executor,
            action_id="act-fail",
            approval_token=token,
            security_code=None,
            action_name="docker.restart",
            action_args={"container": "ipfs_cluster"},
        )
    assert r["http_status"] == 200  # request succeeded; action failed
    line = _last_audit_line(executor)
    assert line["executed"] is True
    assert line["rejected_reason"] == ""
    assert line["result"]["success"] is False
    assert line["result"]["exit_code"] == 1


def test_whitelist_hash_is_content_sensitive(tmp_path):
    """Whitelist hash must change when the file content changes — not
    just be a stable hash of any file. Previous test only proved
    determinism across two reads of the same file."""
    wl_src = _real_whitelist_or_skip()
    p1 = tmp_path / "wl_v1.json"
    p1.write_bytes(wl_src.read_bytes())
    hash_v1 = load_whitelist(str(p1)).sha256_hex
    # Mutate one byte (add a harmless whitespace)
    p2 = tmp_path / "wl_v2.json"
    p2.write_bytes(wl_src.read_bytes() + b" ")
    hash_v2 = load_whitelist(str(p2)).sha256_hex
    assert hash_v1 != hash_v2
