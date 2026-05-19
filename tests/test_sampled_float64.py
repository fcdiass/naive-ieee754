"""Tests for Float64: sampled conversion and sampled arithmetic.

Two layers:
  1. Sampled conversion (500 patterns) — oracle: struct.pack('>d')
  2. Sampled arithmetic  (50 pairs/op) — oracle: Python float (which IS float64)

Float64 arithmetic in the naive library is the slowest of the four formats
(52 significand bits processed bit by bit). The pair count is kept small
on purpose — 50 pairs per operation keeps the suite under a few seconds
while still exercising the full arithmetic pipeline.
"""

from __future__ import annotations

import math
import random
import struct

from naive_ieee754 import Float64

# ---------------------------------------------------------------------------
# Pre-computed data (module level — paid once per session)
# ---------------------------------------------------------------------------


def _is_nan_f64(p: int) -> bool:
    return ((p >> 52) & 0x7FF) == 0x7FF and (p & ((1 << 52) - 1)) != 0


_rng = random.Random(42)

_raw = [_rng.randint(0, 0xFFFFFFFFFFFFFFFF) for _ in range(700)]
_NON_NAN_F64 = [p for p in _raw if not _is_nan_f64(p)]

_CONV_PATTERNS = _NON_NAN_F64[:500]
_ARITH_PATTERNS = _NON_NAN_F64[:200]

_SAMPLE_PAIRS = [
    (_rng.choice(_ARITH_PATTERNS), _rng.choice(_ARITH_PATTERNS))
    for _ in range(50)
]


# ---------------------------------------------------------------------------
# Layer 1: sampled conversion — oracle = struct (Python float IS float64)
# ---------------------------------------------------------------------------


def test_float64_to_decimal_sampled():
    """Sampled Float64 patterns decode to the same value as struct.unpack('>d')."""
    bad = []
    for p in _CONV_PATTERNS:
        ref = struct.unpack(">d", struct.pack(">Q", p))[0]
        got = Float64.from_raw_int(p).to_decimal()
        if math.isinf(ref):
            ok = math.isinf(got) and (got > 0) == (ref > 0)
        elif ref == 0.0:
            ok = got == 0.0 and math.copysign(1.0, got) == math.copysign(1.0, ref)
        else:
            ok = got == ref
        if not ok:
            bad.append((f"{p:#018x}", ref, got))
    assert not bad, f"{len(bad)} to_decimal mismatches; first 3: {bad[:3]}"


def test_float64_from_decimal_sampled():
    """Sampled non-NaN Float64 patterns survive a from_decimal round-trip."""
    bad = []
    for p in _CONV_PATTERNS:
        f = Float64.from_raw_int(p)
        back = Float64.from_decimal(f.to_decimal())
        if back.to_hex() != f.to_hex():
            bad.append((f"{p:#018x}", f.to_hex(), back.to_hex()))
    assert not bad, f"{len(bad)} round-trip mismatches; first 3: {bad[:3]}"


# ---------------------------------------------------------------------------
# Layer 2: sampled arithmetic — oracle = Python float (= IEEE 754 float64)
# ---------------------------------------------------------------------------


def _run_sampled(op_name: str, lib_op, py_op) -> None:
    bad = []
    for pa, pb in _SAMPLE_PAIRS:
        fa = Float64.from_raw_int(pa)
        fb = Float64.from_raw_int(pb)

        if op_name == "div" and fb.is_zero():
            continue  # div-by-zero covered by test_special.py

        lib_res = lib_op(fa, fb)
        a_val = fa.to_decimal()
        b_val = fb.to_decimal()

        try:
            exact = py_op(a_val, b_val)
        except (ZeroDivisionError, ValueError):
            if not (lib_res.is_nan() or lib_res.is_inf()):
                bad.append((fa.to_hex(), fb.to_hex(), "expected Inf/NaN"))
            continue

        if math.isnan(exact):
            if not lib_res.is_nan():
                bad.append((fa.to_hex(), fb.to_hex(), f"expected NaN, got {lib_res.to_hex()}"))
            continue

        if lib_res.is_zero() and exact == 0.0:
            continue  # sign of zero covered by test_special.py

        ref = Float64.from_decimal(exact)
        if ref.to_hex() != lib_res.to_hex():
            bad.append((fa.to_hex(), fb.to_hex(),
                        f"lib={lib_res.to_hex()} ref={ref.to_hex()} exact={exact!r}"))

    assert not bad, (
        f"{len(bad)} mismatches for Float64 {op_name}; first 3:\n"
        + "\n".join(str(e) for e in bad[:3])
    )


def test_float64_add_sampled():
    _run_sampled("add", lambda a, b: a + b, lambda a, b: a + b)


def test_float64_sub_sampled():
    _run_sampled("sub", lambda a, b: a - b, lambda a, b: a - b)


def test_float64_mul_sampled():
    _run_sampled("mul", lambda a, b: a * b, lambda a, b: a * b)


def test_float64_div_sampled():
    _run_sampled("div", lambda a, b: a / b, lambda a, b: a / b)
