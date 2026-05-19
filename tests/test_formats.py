"""Tests for FloatFormat properties."""

from naive_ieee754.formats import (
    FLOAT8_FORMAT,
    FLOAT16_FORMAT,
    FLOAT32_FORMAT,
    FLOAT64_FORMAT,
)


def test_total_bits():
    assert FLOAT8_FORMAT.total_bits == 8
    assert FLOAT16_FORMAT.total_bits == 16
    assert FLOAT32_FORMAT.total_bits == 32
    assert FLOAT64_FORMAT.total_bits == 64


def test_bias():
    assert FLOAT8_FORMAT.bias == 7
    assert FLOAT16_FORMAT.bias == 15
    assert FLOAT32_FORMAT.bias == 127
    assert FLOAT64_FORMAT.bias == 1023


def test_max_stored_exponent():
    assert FLOAT8_FORMAT.max_stored_exponent == 15
    assert FLOAT32_FORMAT.max_stored_exponent == 255
    assert FLOAT64_FORMAT.max_stored_exponent == 2047


def test_max_actual_exponent():
    assert FLOAT32_FORMAT.max_actual_exponent == 127
    assert FLOAT64_FORMAT.max_actual_exponent == 1023


def test_min_actual_exponent_normal():
    assert FLOAT32_FORMAT.min_actual_exponent_normal == -126
    assert FLOAT64_FORMAT.min_actual_exponent_normal == -1022


def test_max_normal_float32():
    expected = (2 - 2**-23) * 2**127
    assert abs(FLOAT32_FORMAT.max_normal - expected) < 1e25


def test_min_positive_normal_float32():
    assert FLOAT32_FORMAT.min_positive_normal == 2**-126


def test_min_positive_subnormal_float32():
    assert FLOAT32_FORMAT.min_positive_subnormal == 2 ** (-126 - 23)


def test_eps_formula():
    """eps must equal 2^(-significand_bits) for every predefined format."""
    assert FLOAT8_FORMAT.eps == 2 ** (-3)
    assert FLOAT16_FORMAT.eps == 2 ** (-10)
    assert FLOAT32_FORMAT.eps == 2 ** (-23)
    assert FLOAT64_FORMAT.eps == 2 ** (-52)


def test_eps_is_one_ulp_of_one():
    """1 + eps must be the next representable value after 1."""
    from naive_ieee754 import Float32, Float64
    from naive_ieee754.bits import bits_to_int

    def next_value(f):
        pat = bits_to_int([f.sign] + f.exponent + f.significand)
        return type(f).from_raw_int(pat + 1)

    one32 = Float32.from_decimal(1.0)
    assert next_value(one32).to_decimal() - 1.0 == FLOAT32_FORMAT.eps

    one64 = Float64.from_decimal(1.0)
    assert next_value(one64).to_decimal() - 1.0 == FLOAT64_FORMAT.eps
