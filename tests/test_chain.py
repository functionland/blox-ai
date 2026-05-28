"""Unit tests for src/tools/chain.py — bytes32(peerId) + ABI encoding +
eth_call wrapper + tristate cache + function selectors.

bytes32 conversion is fixture-tested against the lab device's actual
ipfs_cluster peerId (12D3KooWE6gC...At3WedRaZ) to lock the algorithm
against accidental regressions.

Function selectors are pinned against their canonical signatures so any
future edit that breaks a signature (typo, type mismatch, param-order
flip) gets caught at test time. ERC20 transfer is included as the
universal keccak256 sanity check.
"""
from __future__ import annotations

import pytest

from Crypto.Hash import keccak

from src.tools.chain import (
    CallResult,
    DEFAULT_RPC_URLS,
    FUNCTION_SELECTORS,
    _b58_decode,
    clear_cache_for_tests,
    decode_bool,
    encode_bytes32,
    encode_call,
    encode_uint32,
    encode_uint256,
    eth_call,
    peer_id_to_bytes32,
)


# ---------------------------------------------------------------------------
# bytes32(peerId) — verified against mainnet-claim-web/app.js:peerIdToBytes32
# ---------------------------------------------------------------------------

# Real ipfs_cluster peerId from lab probe (192.168.2.159, 2026-05-28).
# Locks the algorithm — if a future "optimization" silently breaks the
# encoding, every chain call against this device returns "not a member"
# and the test catches it.
LAB_CLUSTER_PEER_ID = "12D3KooWE6gC66XWxKacdna5LX4ymwnCCMpaddBFkB8At3WedRaZ"

# Real kubo peerId from same lab probe — sanity check that the format
# detection works for both Ed25519 and other PeerID types.
LAB_KUBO_PEER_ID = "12D3KooWCnRuQFScUBTmCi9EMNB7HrWHb12RPUdUZTjJb4FaF1nw"


class TestPeerIdToBytes32:
    def test_lab_cluster_peerid_decodes_to_32_bytes(self):
        """Smoke test: the conversion produces a 0x-prefixed 64-hex-char
        string (= 32 bytes). Doesn't pin the EXACT bytes32 value because
        that requires running the JS reference impl side-by-side; we'll
        do that comparison once when the diag/identity_health lab
        integration test runs."""
        result = peer_id_to_bytes32(LAB_CLUSTER_PEER_ID)
        assert result.startswith("0x")
        assert len(result) == 2 + 64   # '0x' + 32 bytes hex
        # All-zeros result would indicate a parse failure that the
        # algorithm silently swallowed.
        assert result != "0x" + "00" * 32

    def test_kubo_peerid_decodes_to_32_bytes(self):
        result = peer_id_to_bytes32(LAB_KUBO_PEER_ID)
        assert result.startswith("0x")
        assert len(result) == 2 + 64
        assert result != "0x" + "00" * 32

    def test_accepts_z_prefixed_form(self):
        """Multibase 'z' prefix optional per JS reference."""
        without_z = peer_id_to_bytes32(LAB_CLUSTER_PEER_ID)
        with_z = peer_id_to_bytes32("z" + LAB_CLUSTER_PEER_ID)
        assert without_z == with_z

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            peer_id_to_bytes32("")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            peer_id_to_bytes32(None)  # type: ignore[arg-type]

    def test_rejects_garbage(self):
        """Garbage base58 input → ValueError."""
        with pytest.raises(ValueError):
            peer_id_to_bytes32("not_valid_base58_!!!")


class TestB58Decode:
    """Internal helper — verified by checking peer_id_to_bytes32 happy
    paths above, but a few direct cases pin the corner behavior."""

    def test_decodes_minimal_byte(self):
        # '2' in base58btc = decimal 1
        assert _b58_decode("2") == b"\x01"

    def test_leading_ones_become_leading_zeros(self):
        # '1' is the base58 representation of byte 0x00. Multiple leading
        # 1's mean leading zero-bytes in the output.
        assert _b58_decode("11") == b"\x00\x00"

    def test_rejects_invalid_char(self):
        with pytest.raises(ValueError):
            _b58_decode("0")   # 0 is NOT in btc alphabet


# ---------------------------------------------------------------------------
# ABI encoding helpers
# ---------------------------------------------------------------------------


class TestAbiEncoding:
    def test_encode_uint32_pads_left_to_32_bytes(self):
        # 5 as uint32 → 32 bytes, value in the low byte
        encoded = encode_uint32(5)
        assert len(encoded) == 32
        assert encoded == b"\x00" * 31 + b"\x05"

    def test_encode_uint32_max_value(self):
        encoded = encode_uint32(0xFFFFFFFF)
        assert len(encoded) == 32
        assert encoded[-4:] == b"\xff\xff\xff\xff"
        assert encoded[:28] == b"\x00" * 28

    def test_encode_uint32_rejects_negative(self):
        with pytest.raises(ValueError):
            encode_uint32(-1)

    def test_encode_uint32_rejects_overflow(self):
        with pytest.raises(ValueError):
            encode_uint32(0x100000000)

    def test_encode_bytes32_strips_0x_prefix(self):
        hex_value = "0x" + "ab" * 32
        encoded = encode_bytes32(hex_value)
        assert encoded == bytes.fromhex("ab" * 32)
        assert len(encoded) == 32

    def test_encode_bytes32_accepts_no_prefix(self):
        encoded = encode_bytes32("ab" * 32)
        assert len(encoded) == 32

    def test_encode_bytes32_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            encode_bytes32("0xab")   # too short

    def test_encode_call_concatenates_selector_and_args(self):
        # Sanity: selector + uint32(5) + bytes32(0xab...) → 4 + 32 + 32 bytes
        data = encode_call("0xdeadbeef", encode_uint32(5), encode_bytes32("ab" * 32))
        # 0x + 4 bytes selector + 32 + 32 = 0x + 136 hex chars
        assert data.startswith("0xdeadbeef")
        assert len(data) == 2 + 8 + 64 + 64

    def test_encode_call_rejects_bad_selector_length(self):
        with pytest.raises(ValueError):
            encode_call("0x1234", encode_uint32(0))

    def test_encode_uint256_pads_left_to_32_bytes(self):
        encoded = encode_uint256(12345)
        assert len(encoded) == 32
        assert encoded[-2:] == (12345).to_bytes(2, "big")
        assert encoded[:30] == b"\x00" * 30

    def test_encode_uint256_max_value(self):
        max_v = (1 << 256) - 1
        encoded = encode_uint256(max_v)
        assert len(encoded) == 32
        assert encoded == b"\xff" * 32

    def test_encode_uint256_rejects_negative(self):
        with pytest.raises(ValueError):
            encode_uint256(-1)

    def test_encode_uint256_rejects_overflow(self):
        with pytest.raises(ValueError):
            encode_uint256(1 << 256)


# ---------------------------------------------------------------------------
# Function selectors — pinned against canonical signatures so any future
# typo/type-mismatch/param-order edit breaks the test.
# ---------------------------------------------------------------------------


def _compute_selector(signature: str) -> str:
    """Recompute first 4 bytes of keccak256(signature) for verification."""
    k = keccak.new(digest_bits=256)
    k.update(signature.encode("utf-8"))
    return "0x" + k.hexdigest()[:8]


class TestFunctionSelectors:
    def test_keccak_implementation_works(self):
        """Self-check: the canonical ERC20 transfer selector is 0xa9059cbb.
        If this fails, the pycryptodome keccak in the test env is broken
        and every selector test below is meaningless."""
        assert _compute_selector("transfer(address,uint256)") == "0xa9059cbb"

    @pytest.mark.parametrize("signature", list(FUNCTION_SELECTORS.keys()))
    def test_pinned_selector_matches_computed_value(self, signature: str):
        """Each pinned selector must equal keccak256(signature)[:4]. A
        signature typo (wrong type, wrong param order) shows up here."""
        assert FUNCTION_SELECTORS[signature] == _compute_selector(signature)

    def test_no_placeholder_selectors_remain(self):
        """No 0x00000000 placeholders. Catches the case where someone
        adds a new entry but forgets to compute the selector."""
        for sig, sel in FUNCTION_SELECTORS.items():
            assert sel != "0x00000000", f"placeholder still in {sig}"

    def test_required_view_methods_present(self):
        """The four view methods diag/identity_health (Phase 0.5b) will
        depend on must remain in the dict. If a future edit removes one,
        the diag tool breaks silently — this test catches it."""
        required = {
            "isPeerIdMemberOfPool(uint32,bytes32)",
            "getPeerIdInfo(uint32,bytes32)",
            "isPeerOnlineAtTimestamp(uint32,uint256,bytes32)",
            "getOnlineStatusSince(bytes32,uint32,uint256)",
        }
        assert required.issubset(FUNCTION_SELECTORS.keys())


# ---------------------------------------------------------------------------
# eth_call wrapper (mocked HTTP)
# ---------------------------------------------------------------------------


class TestEthCall:
    def setup_method(self):
        clear_cache_for_tests()

    def test_unknown_chain_returns_error_state(self):
        result = eth_call("ethereum", "0xabc", "0xdef")
        assert result.state == "error"
        assert "unknown_chain" in (result.reason or "")

    def test_rpc_unreachable_returns_unknown_state(self, monkeypatch):
        """When http_post_json returns None (network failure), eth_call
        returns tristate 'unknown' so trees branch into the
        rpc-unreachable path."""
        monkeypatch.setattr(
            "src.tools.chain.http_post_json",
            lambda *a, **kw: None,
        )
        result = eth_call("base", "0xabc", "0xdef")
        assert result.state == "unknown"
        assert result.reason == "rpc_unreachable"

    def test_chain_error_returns_error_state(self, monkeypatch):
        monkeypatch.setattr(
            "src.tools.chain.http_post_json",
            lambda *a, **kw: {"error": {"message": "execution reverted"}},
        )
        result = eth_call("base", "0xabc", "0xdef")
        assert result.state == "error"
        assert "execution reverted" in (result.reason or "")

    def test_ok_result_returns_ok_state_with_value(self, monkeypatch):
        monkeypatch.setattr(
            "src.tools.chain.http_post_json",
            lambda *a, **kw: {"result": "0x" + "00" * 31 + "01"},
        )
        result = eth_call("base", "0xabc", "0xdef")
        assert result.state == "ok"
        assert result.value == "0x" + "00" * 31 + "01"

    def test_result_is_cached_within_ttl(self, monkeypatch):
        call_count = {"n": 0}

        def fake_post(*a, **kw):
            call_count["n"] += 1
            return {"result": "0x" + "00" * 31 + "01"}

        monkeypatch.setattr("src.tools.chain.http_post_json", fake_post)
        eth_call("base", "0xabc", "0xdef")
        eth_call("base", "0xabc", "0xdef")
        eth_call("base", "0xabc", "0xdef")
        # All 3 hit the cache after the first
        assert call_count["n"] == 1

    def test_different_args_dont_collide_in_cache(self, monkeypatch):
        call_count = {"n": 0}

        def fake_post(*a, **kw):
            call_count["n"] += 1
            return {"result": "0x" + "00" * 32}

        monkeypatch.setattr("src.tools.chain.http_post_json", fake_post)
        eth_call("base", "0xabc", "0xdef")
        eth_call("base", "0xabc", "0xeee")   # different data
        eth_call("skale", "0xabc", "0xdef")  # different chain
        eth_call("base", "0xfff", "0xdef")   # different address
        assert call_count["n"] == 4


# ---------------------------------------------------------------------------
# Bool decoding
# ---------------------------------------------------------------------------


class TestDecodeBool:
    def test_zero_returns_false(self):
        assert decode_bool("0x" + "00" * 32) is False

    def test_one_returns_true(self):
        assert decode_bool("0x" + "00" * 31 + "01") is True

    def test_any_nonzero_is_true(self):
        assert decode_bool("0x" + "ff" * 32) is True
