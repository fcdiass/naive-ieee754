"""Tests for IEEEFloat construction, conversion, and display."""

import math
import struct
import pytest
from naive_ieee754 import (
    Float8,
    Float32,
    Float64,
    IEEEFloat,
    SpecialKind,
    custom_float,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def float32_ground_truth(value: float) -> float:
    """Python's struct round-trip through float32."""
    packed = struct.pack(">f", value)
    return struct.unpack(">f", packed)[0]


# ---------------------------------------------------------------------------
# Special value factories
# ---------------------------------------------------------------------------


def test_positive_zero():
    z = Float32.positive_zero()
    assert z.kind == SpecialKind.POSITIVE_ZERO
    assert z.is_zero()
    assert z.to_decimal() == 0.0
    assert math.copysign(1.0, z.to_decimal()) > 0


def test_negative_zero():
    z = Float32.negative_zero()
    assert z.kind == SpecialKind.NEGATIVE_ZERO
    assert z.is_zero()
    assert math.copysign(1.0, z.to_decimal()) < 0


def test_positive_infinity():
    inf = Float32.positive_infinity()
    assert inf.is_inf()
    assert math.isinf(inf.to_decimal())
    assert inf.to_decimal() > 0


def test_negative_infinity():
    inf = Float32.negative_infinity()
    assert inf.is_inf()
    assert inf.to_decimal() < 0


def test_nan():
    nan = Float32.nan()
    assert nan.is_nan()
    assert math.isnan(nan.to_decimal())


def test_positive_zero_equals_negative_zero():
    assert Float32.positive_zero() == Float32.negative_zero()


def test_nan_not_equal_to_nan():
    assert Float32.nan() != Float32.nan()


# ---------------------------------------------------------------------------
# from_decimal round-trip against struct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        -1.0,
        1.5,
        -1.5,
        0.5,
        -0.5,
        2.0,
        0.25,
        100.0,
        -100.0,
        1234.5,
        -1234.5,
        1e10,
        -1e10,
        1e30,
        -1e30,
        3.14159265358979,
    ],
)
def test_float32_round_trip(value):
    f = Float32.from_decimal(value)
    expected = float32_ground_truth(value)
    assert f.to_decimal() == expected, (
        f"from_decimal({value}) gave {f.to_decimal()!r}, expected {expected!r}"
    )


def test_from_decimal_zero():
    assert Float32.from_decimal(0.0).is_zero()
    assert Float32.from_decimal(-0.0).kind == SpecialKind.NEGATIVE_ZERO


def test_from_decimal_nan():
    assert Float32.from_decimal(float("nan")).is_nan()


def test_from_decimal_inf():
    assert Float32.from_decimal(float("inf")).kind == SpecialKind.POSITIVE_INF
    assert Float32.from_decimal(float("-inf")).kind == SpecialKind.NEGATIVE_INF


def test_from_decimal_overflow_to_inf():
    # A value larger than Float8 max normal should give +Inf
    f = Float8.from_decimal(1e38)
    assert f.kind == SpecialKind.POSITIVE_INF


@pytest.mark.parametrize(
    "bits",
    [
        0x00000001,  # smallest positive subnormal
        0x007FFFFF,  # largest positive subnormal
        0x00800000,  # smallest positive normal
        0x2EDBE6FF,  # 1e-10 as float32
        0x1E3CE508,  # 1e-20 as float32
    ],
)
def test_float32_from_decimal_matches_reference_bits(bits):
    value = struct.unpack(">f", bits.to_bytes(4, "big"))[0]
    assert Float32.from_decimal(value).to_hex() == f"{bits:08X}"


def test_subnormal_to_decimal():
    f = Float32.from_raw_int(0x00000001)
    assert f.kind == SpecialKind.SUBNORMAL
    assert f.to_decimal() == 2**-149


def test_from_decimal_large_int_does_not_require_python_float_conversion():
    f = Float32.from_decimal(2**200)
    assert f.kind == SpecialKind.POSITIVE_INF


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_to_bit_string_format():
    f = Float32.from_decimal(1.0)
    s = f.to_bit_string()
    parts = s.split(" ")
    assert len(parts) == 3
    assert len(parts[0]) == 1  # sign
    assert len(parts[1]) == 8  # exponent
    assert len(parts[2]) == 23  # significand


def test_to_hex_length():
    assert len(Float32.from_decimal(1.0).to_hex()) == 8  # 32 bits = 8 hex chars
    assert len(Float64.from_decimal(1.0).to_hex()) == 16  # 64 bits = 16 hex chars
    assert len(Float8.from_decimal(1.0).to_hex()) == 2  # 8 bits = 2 hex chars


def test_repr_contains_format_name():
    r = repr(Float32.from_decimal(1.0))
    assert "Float32" in r
    assert "sign=" in r
    assert "exp=" in r
    assert "sig=" in r


def test_str_contains_decimal():
    s = str(Float32.from_decimal(1.5))
    assert "1.5" in s
    assert "Float32" in s


# ---------------------------------------------------------------------------
# from_bits and from_raw_int
# ---------------------------------------------------------------------------


def test_from_bits_round_trip():
    f = Float32.from_decimal(3.14)
    bits = [f.sign] + f.exponent + f.significand
    f2 = Float32.from_bits(bits)
    assert f == f2


def test_from_raw_int_round_trip():
    f = Float32.from_decimal(2.71828)
    from naive_ieee754.bits import bits_to_int

    pattern = bits_to_int([f.sign] + f.exponent + f.significand)
    f2 = Float32.from_raw_int(pattern)
    assert f == f2


def test_from_bit_string_matches_from_decimal():
    """from_bit_string and from_decimal must agree on the same value."""
    f_dec = Float32.from_decimal(1.5)
    bit_str = f_dec.to_bit_string()
    f_bits = Float32.from_bit_string(bit_str)
    assert f_dec == f_bits


def test_from_bit_string_ignores_spaces():
    """Spaces in the bit string must be silently ignored."""
    f_compact = Float32.from_bit_string("00111111110000000000000000000000")
    f_spaced = Float32.from_bit_string("0 01111111 10000000000000000000000")
    assert f_compact == f_spaced


def test_from_bit_string_round_trip():
    """to_bit_string → from_bit_string must be a no-op for any value."""
    for value in (0.0, -0.0, 1.0, -1.5, 0.1):
        f = Float32.from_decimal(value)
        assert Float32.from_bit_string(f.to_bit_string()) == f


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def test_neg():
    f = Float32.from_decimal(1.5)
    assert (-f).sign == 1
    assert (-f).to_decimal() == -1.5


def test_abs():
    f = Float32.from_decimal(-2.0)
    assert abs(f).sign == 0
    assert abs(f).to_decimal() == 2.0


# ---------------------------------------------------------------------------
# custom_float
# ---------------------------------------------------------------------------


def test_custom_float_is_a_class():
    Mini = custom_float(exponent_bits=4, significand_bits=4)
    assert isinstance(Mini, type)
    assert issubclass(Mini, IEEEFloat)


def test_custom_float_format():
    Mini = custom_float(exponent_bits=4, significand_bits=4)
    assert Mini.FORMAT.total_bits == 9
    assert Mini.FORMAT.bias == 7


def test_custom_float_from_decimal():
    Mini = custom_float(exponent_bits=4, significand_bits=4)
    f = Mini.from_decimal(1.5)
    assert f.is_normal()
    assert f.to_decimal() == 1.5


def test_custom_float_direct_construction():
    Mini = custom_float(exponent_bits=4, significand_bits=4)
    f = Mini.from_decimal(1.5)
    # Direct construction with sign/exponent/significand — no fmt argument
    f2 = Mini(f.sign, f.exponent, f.significand)
    assert f2.to_decimal() == 1.5


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_immutable():
    f = Float32.from_decimal(1.0)
    with pytest.raises(AttributeError):
        f.sign = 1
