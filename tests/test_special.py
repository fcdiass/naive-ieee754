"""Tests for SpecialKind classification."""

from naive_ieee754.special import SpecialKind, classify


def _exp(n, all_ones=False, all_zeros=False):
    if all_ones:
        return [1] * n
    if all_zeros:
        return [0] * n
    return [0] * (n - 1) + [1]  # just the LSB set


def _mant(n, all_zeros=False):
    if all_zeros:
        return [0] * n
    return [0] * (n - 1) + [1]


def test_positive_inf():
    assert (
        classify(0, _exp(8, all_ones=True), _mant(23, all_zeros=True))
        == SpecialKind.POSITIVE_INF
    )


def test_negative_inf():
    assert (
        classify(1, _exp(8, all_ones=True), _mant(23, all_zeros=True))
        == SpecialKind.NEGATIVE_INF
    )


def test_nan():
    assert classify(0, _exp(8, all_ones=True), _mant(23)) == SpecialKind.NAN
    assert classify(1, _exp(8, all_ones=True), _mant(23)) == SpecialKind.NAN


def test_positive_zero():
    assert (
        classify(0, _exp(8, all_zeros=True), _mant(23, all_zeros=True))
        == SpecialKind.POSITIVE_ZERO
    )


def test_negative_zero():
    assert (
        classify(1, _exp(8, all_zeros=True), _mant(23, all_zeros=True))
        == SpecialKind.NEGATIVE_ZERO
    )


def test_subnormal():
    assert classify(0, _exp(8, all_zeros=True), _mant(23)) == SpecialKind.SUBNORMAL


def test_normal():
    assert classify(0, _exp(8), _mant(23)) == SpecialKind.NORMAL
    assert classify(1, _exp(8), _mant(23)) == SpecialKind.NORMAL
