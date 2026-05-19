"""Tests for step-by-step arithmetic operations."""

import math
import struct
import pytest
from naive_ieee754 import (
    Float8,
    Float32,
    IEEEFloat,
    SpecialKind,
    FloatingPointFlag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def f8(value: float) -> IEEEFloat:
    return Float8.from_decimal(value)


def f32(value: float) -> IEEEFloat:
    return Float32.from_decimal(value)


def float32_ground_truth(value: float) -> float:
    packed = struct.pack(">f", value)
    return struct.unpack(">f", packed)[0]


def float32_hex(value: float) -> str:
    return struct.pack(">f", value).hex().upper()


# ---------------------------------------------------------------------------
# Basic addition
# ---------------------------------------------------------------------------


def test_add_simple():
    a = f8(1.0)
    b = f8(0.5)
    result = a + b
    assert result.to_decimal() == 1.5


def test_add_negative():
    a = f32(3.0)
    b = f32(-1.0)
    result = a + b
    assert result.to_decimal() == 2.0


def test_add_zero():
    a = f32(1.5)
    z = Float32.positive_zero()
    assert (a + z).to_decimal() == 1.5
    assert (z + a).to_decimal() == 1.5


def test_add_cancellation():
    a = f32(1.0)
    b = f32(-1.0)
    result = a + b
    assert result.is_zero()


# ---------------------------------------------------------------------------
# Basic subtraction
# ---------------------------------------------------------------------------


def test_sub_simple():
    a = f8(2.0)
    b = f8(0.5)
    result = a - b
    assert result.to_decimal() == 1.5


def test_sub_to_negative():
    a = f32(1.0)
    b = f32(2.0)
    result = a - b
    assert result.to_decimal() == -1.0


# ---------------------------------------------------------------------------
# Basic multiplication
# ---------------------------------------------------------------------------


def test_mul_simple():
    a = f8(2.0)
    b = f8(3.0)
    result = a * b
    assert result.to_decimal() == 6.0


def test_mul_by_zero():
    a = f32(1e10)
    z = Float32.positive_zero()
    assert (a * z).is_zero()


def test_mul_sign():
    a = f32(2.0)
    b = f32(-3.0)
    result = a * b
    assert result.sign == 1
    assert result.to_decimal() == -6.0


# ---------------------------------------------------------------------------
# Basic division
# ---------------------------------------------------------------------------


def test_div_simple():
    a = f8(4.0)
    b = f8(2.0)
    result = a / b
    assert result.to_decimal() == 2.0


def test_div_by_zero():
    a = f32(1.0)
    b = Float32.positive_zero()
    result = a / b
    assert result.kind == SpecialKind.POSITIVE_INF


def test_div_zero_by_zero():
    a = Float32.positive_zero()
    b = Float32.positive_zero()
    assert (a / b).is_nan()


# ---------------------------------------------------------------------------
# Special value propagation
# ---------------------------------------------------------------------------


def test_nan_plus_anything_is_nan():
    nan = Float32.nan()
    a = f32(1.0)
    assert (nan + a).is_nan()
    assert (a + nan).is_nan()


def test_inf_plus_finite_is_inf():
    inf = Float32.positive_infinity()
    a = f32(1e10)
    assert (inf + a).kind == SpecialKind.POSITIVE_INF


def test_inf_minus_inf_is_nan():
    pos_inf = Float32.positive_infinity()
    neg_inf = Float32.negative_infinity()
    assert (pos_inf + neg_inf).is_nan()


def test_zero_times_inf_is_nan():
    z = Float32.positive_zero()
    inf = Float32.positive_infinity()
    assert (z * inf).is_nan()


def test_inf_div_inf_is_nan():
    inf = Float32.positive_infinity()
    assert (inf / inf).is_nan()


# ---------------------------------------------------------------------------
# ArithmeticResult (verbose interface)
# ---------------------------------------------------------------------------


def test_verbose_add_has_steps():
    a = f32(1.5)
    b = f32(2.5)
    result = a.add(b)
    step_names = {s.name for s in result.steps}
    assert "input" in step_names
    assert "special_check" in step_names
    assert "result" in step_names


def test_verbose_mul_has_steps():
    a = f32(2.0)
    b = f32(3.0)
    result = a.mul(b)
    step_names = {s.name for s in result.steps}
    assert "sign" in step_names
    assert "exponent_add" in step_names
    assert "significand_multiply" in step_names


def test_verbose_div_has_steps():
    a = f32(6.0)
    b = f32(2.0)
    result = a.div(b)
    step_names = {s.name for s in result.steps}
    assert "exponent_subtract" in step_names
    assert "significand_divide" in step_names


def test_explain_returns_string():
    result = f32(0.1).add(f32(0.2))
    explanation = result.explain()
    assert isinstance(explanation, str)
    assert "Operation: add" in explanation


def test_precision_report_detects_error():
    # 0.1 + 0.2 is the canonical example of Float32 rounding error
    result = f32(0.1).add(f32(0.2))
    report = result.precision_report()
    assert "exact" in report
    assert "represented" in report


def test_precision_lost_flag():
    # 0.1 cannot be represented exactly in Float32; addition should detect lost precision
    result = f32(0.1).add(f32(0.2))
    assert result.precision_lost


@pytest.mark.parametrize(
    ("op", "a", "b"),
    [
        ("add", 0.1, 0.2),
        ("sub", 0.3, 0.2),
        ("mul", 0.1, 0.2),
        ("div", 1.0, 3.0),
    ],
)
def test_float32_arithmetic_matches_reference_bits(op, a, b):
    lhs = f32(a)
    rhs = f32(b)
    result = {
        "add": lhs + rhs,
        "sub": lhs - rhs,
        "mul": lhs * rhs,
        "div": lhs / rhs,
    }[op]
    expected = {
        "add": float32_ground_truth(float32_ground_truth(a) + float32_ground_truth(b)),
        "sub": float32_ground_truth(float32_ground_truth(a) - float32_ground_truth(b)),
        "mul": float32_ground_truth(float32_ground_truth(a) * float32_ground_truth(b)),
        "div": float32_ground_truth(float32_ground_truth(a) / float32_ground_truth(b)),
    }[op]
    assert result.to_hex() == float32_hex(expected)


def test_subnormal_arithmetic_is_supported():
    tiny = Float32.from_raw_int(0x00000001)
    result = tiny + tiny
    assert result.kind == SpecialKind.SUBNORMAL
    assert result.to_hex() == "00000002"


def test_zero_sign_rules():
    assert (
        Float32.negative_zero() + Float32.negative_zero()
    ).kind == SpecialKind.NEGATIVE_ZERO
    assert (
        Float32.positive_zero() + Float32.negative_zero()
    ).kind == SpecialKind.POSITIVE_ZERO
    assert (
        Float32.from_decimal(1.0) * Float32.negative_zero()
    ).kind == SpecialKind.NEGATIVE_ZERO
    assert (
        Float32.positive_zero() / Float32.from_decimal(-1.0)
    ).kind == SpecialKind.NEGATIVE_ZERO


def test_invalid_flag():
    result = Float32.positive_zero().mul(Float32.positive_infinity())
    assert result.result.is_nan()
    assert FloatingPointFlag.INVALID in result.flags


def test_divide_by_zero_flag():
    result = Float32.from_decimal(1.0).div(Float32.positive_zero())
    assert result.result.kind == SpecialKind.POSITIVE_INF
    assert FloatingPointFlag.DIVIDE_BY_ZERO in result.flags


def test_inexact_flag():
    result = f32(1.0).div(f32(3.0))
    assert FloatingPointFlag.INEXACT in result.flags
    assert result.precision_lost


def test_underflow_flag():
    tiny = Float32.from_raw_int(0x00000001)
    result = tiny.div(f32(3.0))
    assert result.result.is_zero()
    assert FloatingPointFlag.UNDERFLOW in result.flags
    assert FloatingPointFlag.INEXACT in result.flags


# ---------------------------------------------------------------------------
# Mixed-format error
# ---------------------------------------------------------------------------


def test_mixed_format_raises():
    a = Float32.from_decimal(1.0)
    b = Float8.from_decimal(1.0)
    with pytest.raises(ValueError, match="Cannot mix formats"):
        _ = a + b


# ---------------------------------------------------------------------------
# sqrt — special cases
# ---------------------------------------------------------------------------


def test_sqrt_nan_input():
    result = Float32.nan().sqrt()
    assert result.result.is_nan()
    assert FloatingPointFlag.INVALID in result.flags


def test_sqrt_negative_is_invalid():
    result = Float32.from_decimal(-1.0).sqrt()
    assert result.result.is_nan()
    assert FloatingPointFlag.INVALID in result.flags


def test_sqrt_positive_infinity():
    result = Float32.positive_infinity().sqrt()
    assert result.result.kind == SpecialKind.POSITIVE_INF
    assert not result.flags


def test_sqrt_positive_zero():
    result = Float32.positive_zero().sqrt()
    assert result.result.kind == SpecialKind.POSITIVE_ZERO


def test_sqrt_negative_zero():
    result = Float32.negative_zero().sqrt()
    assert result.result.kind == SpecialKind.NEGATIVE_ZERO


# ---------------------------------------------------------------------------
# sqrt — exact results (perfect squares)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", [1.0, 4.0, 9.0, 16.0, 0.25, 0.0625])
def test_sqrt_exact_perfect_squares(val):
    result = Float32.from_decimal(val).sqrt()

    assert result.result.to_decimal() == math.sqrt(val)
    assert FloatingPointFlag.INEXACT not in result.flags


# ---------------------------------------------------------------------------
# sqrt — correctly-rounded results (bit-accurate against struct oracle)
# ---------------------------------------------------------------------------


def float32_sqrt_reference(val: float) -> str:
    """Correctly-rounded float32 sqrt via float64 intermediate (no double-rounding risk
    for sqrt: the exact result is irrational, so float64 precision is more than enough)."""
    import struct

    f32_val = struct.unpack(">f", struct.pack(">f", val))[0]
    return struct.pack(">f", math.sqrt(f32_val)).hex().upper()


@pytest.mark.parametrize("val", [2.0, 3.0, 0.5, 0.1, 7.0, 1e-10, 1e20, 1.23456789])
def test_sqrt_float32_matches_reference_bits(val):
    result = Float32.from_decimal(val).sqrt()
    assert result.result.to_hex() == float32_sqrt_reference(val), (
        f"sqrt({val}): got {result.result.to_hex()}, expected {float32_sqrt_reference(val)}"
    )
    assert FloatingPointFlag.INEXACT in result.flags


# ---------------------------------------------------------------------------
# sqrt — verbose interface
# ---------------------------------------------------------------------------


def test_sqrt_has_steps():
    result = Float32.from_decimal(2.0).sqrt()
    step_names = {s.name for s in result.steps}
    assert "input" in step_names
    assert "integer_sqrt" in step_names
    assert "round" in step_names
    assert "result" in step_names


def test_sqrt_inexact_flag():
    result = Float32.from_decimal(2.0).sqrt()
    assert result.precision_lost
    assert FloatingPointFlag.INEXACT in result.flags


# ---------------------------------------------------------------------------
# sqrt — Float8 exhaustive against math.sqrt oracle
# ---------------------------------------------------------------------------


def test_sqrt_float8_exhaustive():
    """All 256 Float8 patterns produce a correctly-rounded sqrt."""

    bad = []
    for p in range(256):
        f = Float8.from_raw_int(p)
        result = f.sqrt()

        if f.is_nan() or (f.sign == 1 and not f.is_zero()):
            if not result.result.is_nan():
                bad.append((f.to_hex(), "expected NaN"))
            continue
        if f.is_inf():
            if result.result.kind != SpecialKind.POSITIVE_INF:
                bad.append((f.to_hex(), "expected +Inf"))
            continue
        if f.is_zero():
            if not result.result.is_zero():
                bad.append((f.to_hex(), "expected zero"))
            continue

        # Finite positive: compare against Float8.from_decimal(math.sqrt(exact_value))
        exact_val = f.to_decimal()
        ref = Float8.from_decimal(math.sqrt(exact_val))
        if result.result.to_hex() != ref.to_hex():
            bad.append(
                (f.to_hex(), f"got {result.result.to_hex()} expected {ref.to_hex()}")
            )

    assert not bad, f"{len(bad)} sqrt mismatches; first 3: {bad[:3]}"
