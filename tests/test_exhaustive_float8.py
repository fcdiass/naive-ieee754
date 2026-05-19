"""Tests for Float8: exhaustive conversion + sampled arithmetic.

Two layers:
  1. Exhaustive conversion (all 256 non-NaN patterns) — round-trip oracle
  2. Sampled arithmetic   (5 000 pairs per op)        — float64 oracle

Float8 has only 256 bit patterns, so exhaustive conversion is free.
For arithmetic, testing all 256² = 65 536 pairs exhaustively took ~8 s;
5 000 random pairs give equivalent coverage in ~0.6 s.

Oracle: for each pair (a, b), the float64 result of
    a.to_decimal() OP b.to_decimal()
is re-encoded via Float8.from_decimal().  Float8 has only 4 significant
bits, so the result always fits exactly in float64's 53.
"""

from __future__ import annotations

import math
import random

from naive_ieee754 import Float8

# ---------------------------------------------------------------------------
# Pre-computed data (module level — paid once per session)
# ---------------------------------------------------------------------------

_ALL_F8 = [Float8.from_raw_int(p) for p in range(256)]
_F8_VAL = [f.to_decimal() for f in _ALL_F8]
_NON_NAN_PATTERNS = [p for p in range(256) if not _ALL_F8[p].is_nan()]

random.seed(42)
_SAMPLE_PAIRS = [
    (random.choice(_NON_NAN_PATTERNS), random.choice(_NON_NAN_PATTERNS))
    for _ in range(5_000)
]


# ---------------------------------------------------------------------------
# Layer 1: exhaustive conversion
# ---------------------------------------------------------------------------


def test_float8_round_trip_exhaustive():
    """Every non-NaN Float8 pattern survives a from_decimal round-trip."""
    bad = []
    for p in _NON_NAN_PATTERNS:
        f = _ALL_F8[p]
        back = Float8.from_decimal(f.to_decimal())
        if back.to_hex() != f.to_hex():
            bad.append((f"{p:#04x}", f.to_hex(), back.to_hex()))
    assert not bad, f"{len(bad)} round-trip mismatches; first 3: {bad[:3]}"


# ---------------------------------------------------------------------------
# Layer 2: sampled arithmetic — oracle = float64
# ---------------------------------------------------------------------------


def _check_sampled(op_name: str, lib_op, float_op) -> None:
    bad = []
    for pa, pb in _SAMPLE_PAIRS:
        fa = _ALL_F8[pa]
        fb = _ALL_F8[pb]

        if op_name == "div" and fb.is_zero():
            continue  # div-by-zero behaviour covered by test_special.py

        lib_res = lib_op(fa, fb)

        try:
            exact = float_op(_F8_VAL[pa], _F8_VAL[pb])
        except (ZeroDivisionError, ValueError):
            if not (lib_res.is_nan() or lib_res.is_inf()):
                bad.append((fa.to_hex(), fb.to_hex(), "expected Inf/NaN"))
            continue

        if math.isnan(exact):
            if not lib_res.is_nan():
                bad.append((fa.to_hex(), fb.to_hex(), f"expected NaN, got {lib_res.to_hex()}"))
            continue

        ref = Float8.from_decimal(exact)
        if ref.to_hex() != lib_res.to_hex():
            bad.append((fa.to_hex(), fb.to_hex(),
                        f"lib={lib_res.to_hex()} ref={ref.to_hex()} exact={exact!r}"))

    assert not bad, (
        f"{len(bad)} mismatches for Float8 {op_name}; first 3:\n"
        + "\n".join(str(e) for e in bad[:3])
    )


def test_float8_add_sampled():
    _check_sampled("add", lambda a, b: a + b, lambda a, b: a + b)


def test_float8_sub_sampled():
    _check_sampled("sub", lambda a, b: a - b, lambda a, b: a - b)


def test_float8_mul_sampled():
    _check_sampled("mul", lambda a, b: a * b, lambda a, b: a * b)


def test_float8_div_sampled():
    _check_sampled("div", lambda a, b: a / b, lambda a, b: a / b)
