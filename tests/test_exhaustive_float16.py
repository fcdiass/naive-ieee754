"""Exhaustive and sampled tests for Float16, using struct and numpy as independent oracles.

Two layers of validation:
  1. Exhaustive conversion (all 65 536 patterns) — oracle: struct.pack('>e')
  2. Sampled arithmetic  (10 000 pairs per op) — oracle: numpy.float16

Layer 1 validates pack_rational and to_decimal independently of the library's
own arithmetic. Layer 2 validates that the arithmetic pipeline produces the
same bit pattern as a well-known IEEE 754-compliant implementation.
"""

from __future__ import annotations

import math
import random
import struct

import numpy as np

from naive_ieee754 import Float16

# ---------------------------------------------------------------------------
# Pre-computed data (module level — paid once per test session)
# ---------------------------------------------------------------------------

_ALL_F16 = [Float16.from_raw_int(p) for p in range(65536)]
_NON_NAN_PATTERNS = [p for p in range(65536) if not _ALL_F16[p].is_nan()]

random.seed(42)
_SAMPLE_PAIRS = [
    (random.choice(_NON_NAN_PATTERNS), random.choice(_NON_NAN_PATTERNS))
    for _ in range(10_000)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _np_to_hex(x: np.float16) -> str:
    return struct.pack(">e", float(x)).hex().upper()


# ---------------------------------------------------------------------------
# Layer 1: exhaustive conversion — oracle = struct
# ---------------------------------------------------------------------------


def test_float16_to_decimal_exhaustive():
    """Every Float16 bit pattern decodes to the same float as struct.unpack('>e')."""
    bad = []
    for p in _NON_NAN_PATTERNS:
        ref = struct.unpack(">e", struct.pack(">H", p))[0]
        got = _ALL_F16[p].to_decimal()
        if math.isinf(ref):
            ok = math.isinf(got) and (got > 0) == (ref > 0)
        elif ref == 0.0:
            ok = got == 0.0 and math.copysign(1.0, got) == math.copysign(1.0, ref)
        else:
            ok = got == ref
        if not ok:
            bad.append((f"{p:#06x}", ref, got))
    assert not bad, f"{len(bad)} to_decimal mismatches; first 3: {bad[:3]}"


def test_float16_from_decimal_exhaustive():
    """Every non-NaN Float16 pattern survives a from_decimal round-trip."""
    bad = []
    for p in _NON_NAN_PATTERNS:
        f = _ALL_F16[p]
        back = Float16.from_decimal(f.to_decimal())
        if back.to_hex() != f.to_hex():
            bad.append((f"{p:#06x}", f.to_hex(), back.to_hex()))
    assert not bad, f"{len(bad)} round-trip mismatches; first 3: {bad[:3]}"


# ---------------------------------------------------------------------------
# Layer 2: sampled arithmetic — oracle = numpy.float16
# ---------------------------------------------------------------------------


def _run_sampled(op_name: str, lib_op, np_op) -> None:
    bad = []
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for pa, pb in _SAMPLE_PAIRS:
            fa = _ALL_F16[pa]
            fb = _ALL_F16[pb]

            if op_name == "div" and fb.is_zero():
                continue  # div-by-zero: behaviour covered by unit tests

            lib_res = lib_op(fa, fb)
            np_res = np_op(
                np.float16(fa.to_decimal()),
                np.float16(fb.to_decimal()),
            )

            np_nan = bool(np.isnan(np_res))
            lib_nan = lib_res.is_nan()
            if np_nan and lib_nan:
                continue
            if np_nan or lib_nan:
                bad.append(
                    (
                        fa.to_hex(),
                        fb.to_hex(),
                        "NaN mismatch",
                        f"lib_nan={lib_nan} np_nan={np_nan}",
                    )
                )
                continue

            # Both zero: sign rules are covered by dedicated unit tests
            if lib_res.is_zero() and float(np_res) == 0.0:
                continue

            expected = _np_to_hex(np_res)
            if lib_res.to_hex() != expected:
                bad.append(
                    (
                        fa.to_hex(),
                        fb.to_hex(),
                        f"lib={lib_res.to_hex()} numpy={expected}",
                    )
                )

    assert not bad, f"{len(bad)} mismatches for Float16 {op_name}; first 3: {bad[:3]}"


def test_float16_add_sampled():
    _run_sampled("add", lambda a, b: a + b, lambda a, b: a + b)


def test_float16_sub_sampled():
    _run_sampled("sub", lambda a, b: a - b, lambda a, b: a - b)


def test_float16_mul_sampled():
    _run_sampled("mul", lambda a, b: a * b, lambda a, b: a * b)


def test_float16_div_sampled():
    _run_sampled("div", lambda a, b: a / b, lambda a, b: a / b)
