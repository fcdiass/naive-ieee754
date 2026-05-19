"""Tests for Float32: sampled conversion and sampled arithmetic.

Two layers:
  1. Sampled conversion (2 000 patterns) — oracle: struct.pack('>f')
  2. Sampled arithmetic   (500 pairs/op) — oracle: numpy.float32

Float32 has ~4 billion bit patterns — exhaustive testing is not feasible.
Both layers use random sampling seeded for reproducibility.
"""

from __future__ import annotations

import math
import random
import struct

import numpy as np

from naive_ieee754 import Float32

# ---------------------------------------------------------------------------
# Pre-computed data (module level — paid once per session)
# ---------------------------------------------------------------------------


def _is_nan_f32(p: int) -> bool:
    return ((p >> 23) & 0xFF) == 0xFF and (p & 0x7FFFFF) != 0


_rng = random.Random(42)

# Draw enough raw patterns and discard the rare NaN ones.
_raw = [_rng.randint(0, 0xFFFFFFFF) for _ in range(3_000)]
_NON_NAN_F32 = [p for p in _raw if not _is_nan_f32(p)]

_CONV_PATTERNS = _NON_NAN_F32[:2_000]
_ARITH_PATTERNS = _NON_NAN_F32[:1_000]  # subset used for arithmetic pairs

_SAMPLE_PAIRS = [
    (_rng.choice(_ARITH_PATTERNS), _rng.choice(_ARITH_PATTERNS))
    for _ in range(500)
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _np_to_hex(x: np.float32) -> str:
    return struct.pack(">f", float(x)).hex().upper()


# ---------------------------------------------------------------------------
# Layer 1: sampled conversion — oracle = struct
# ---------------------------------------------------------------------------


def test_float32_to_decimal_sampled():
    """Sampled Float32 patterns decode to the same value as struct.unpack('>f')."""
    bad = []
    for p in _CONV_PATTERNS:
        ref = struct.unpack(">f", struct.pack(">I", p))[0]
        got = Float32.from_raw_int(p).to_decimal()
        if math.isinf(ref):
            ok = math.isinf(got) and (got > 0) == (ref > 0)
        elif ref == 0.0:
            ok = got == 0.0 and math.copysign(1.0, got) == math.copysign(1.0, ref)
        else:
            ok = got == ref
        if not ok:
            bad.append((f"{p:#010x}", ref, got))
    assert not bad, f"{len(bad)} to_decimal mismatches; first 3: {bad[:3]}"


def test_float32_from_decimal_sampled():
    """Sampled non-NaN Float32 patterns survive a from_decimal round-trip."""
    bad = []
    for p in _CONV_PATTERNS:
        f = Float32.from_raw_int(p)
        back = Float32.from_decimal(f.to_decimal())
        if back.to_hex() != f.to_hex():
            bad.append((f"{p:#010x}", f.to_hex(), back.to_hex()))
    assert not bad, f"{len(bad)} round-trip mismatches; first 3: {bad[:3]}"


# ---------------------------------------------------------------------------
# Layer 2: sampled arithmetic — oracle = numpy.float32
# ---------------------------------------------------------------------------


def _run_sampled(op_name: str, lib_op, np_op) -> None:
    bad = []
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for pa, pb in _SAMPLE_PAIRS:
            fa = Float32.from_raw_int(pa)
            fb = Float32.from_raw_int(pb)

            if op_name == "div" and fb.is_zero():
                continue  # div-by-zero covered by test_special.py

            lib_res = lib_op(fa, fb)
            np_res = np_op(np.float32(fa.to_decimal()), np.float32(fb.to_decimal()))

            np_nan = bool(np.isnan(np_res))
            lib_nan = lib_res.is_nan()
            if np_nan and lib_nan:
                continue
            if np_nan or lib_nan:
                bad.append((fa.to_hex(), fb.to_hex(), f"NaN mismatch lib={lib_nan} np={np_nan}"))
                continue

            if lib_res.is_zero() and float(np_res) == 0.0:
                continue  # sign of zero covered by test_special.py

            expected = _np_to_hex(np_res)
            if lib_res.to_hex() != expected:
                bad.append((fa.to_hex(), fb.to_hex(),
                             f"lib={lib_res.to_hex()} numpy={expected}"))

    assert not bad, f"{len(bad)} mismatches for Float32 {op_name}; first 3: {bad[:3]}"


def test_float32_add_sampled():
    _run_sampled("add", lambda a, b: a + b, lambda a, b: a + b)


def test_float32_sub_sampled():
    _run_sampled("sub", lambda a, b: a - b, lambda a, b: a - b)


def test_float32_mul_sampled():
    _run_sampled("mul", lambda a, b: a * b, lambda a, b: a * b)


def test_float32_div_sampled():
    _run_sampled("div", lambda a, b: a / b, lambda a, b: a / b)
