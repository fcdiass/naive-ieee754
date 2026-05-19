"""Exact binary helpers for IEEE 754 encoding and arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from .bits import bits_to_int, int_to_bits
from .flags import FloatingPointFlag
from .formats import FloatFormat
from .rounding import RoundingMode
from .special import SpecialKind

# Registry: FloatFormat → concrete IEEEFloat subclass.
# Populated automatically by IEEEFloat.__init_subclass__ when a subclass is defined.
_FMT_TO_CLASS: dict = {}


def _instantiate(sign: int, exponent: list, significand: list, fmt: FloatFormat) -> object:
    """Create an IEEEFloat instance of the concrete class registered for fmt.

    Every FloatFormat used at runtime must have a concrete IEEEFloat subclass
    registered (this happens automatically via IEEEFloat.__init_subclass__).
    """
    cls = _FMT_TO_CLASS.get(fmt)
    if cls is None:
        raise LookupError(
            f"No IEEEFloat subclass is registered for format {fmt.name!r}. "
            f"Create one with custom_float() or subclass IEEEFloat directly."
        )
    return cls(sign, exponent, significand)


def _make_zero(sign: int, fmt: FloatFormat) -> object:
    """Internal helper: construct ±0 in the given format."""
    return _instantiate(sign, [0] * fmt.exponent_bits, [0] * fmt.significand_bits, fmt)


def _make_inf(sign: int, fmt: FloatFormat) -> object:
    """Internal helper: construct ±Inf in the given format."""
    return _instantiate(sign, [1] * fmt.exponent_bits, [0] * fmt.significand_bits, fmt)


def _make_nan(fmt: FloatFormat) -> object:
    """Internal helper: construct a canonical quiet NaN in the given format."""
    significand = [0] * fmt.significand_bits
    if fmt.significand_bits > 0:
        significand[0] = 1
    return _instantiate(0, [1] * fmt.exponent_bits, significand, fmt)


@dataclass(frozen=True)
class FiniteParts:
    """A finite value represented as significand * 2**exponent."""

    sign: int
    significand: int
    exponent: int


def decode_finite(value) -> FiniteParts:
    """Decode a finite IEEEFloat into an exact integer significand and exponent."""
    kind = value.kind
    if kind in (SpecialKind.NAN, SpecialKind.POSITIVE_INF, SpecialKind.NEGATIVE_INF):
        raise ValueError("Cannot decode a non-finite value.")

    p = value.fmt.significand_bits
    sig_bits = bits_to_int(value.significand)

    if kind in (SpecialKind.POSITIVE_ZERO, SpecialKind.NEGATIVE_ZERO):
        return FiniteParts(value.sign, 0, value.fmt.min_actual_exponent_normal - p)

    if kind == SpecialKind.SUBNORMAL:
        return FiniteParts(
            value.sign,
            sig_bits,
            value.fmt.min_actual_exponent_normal - p,
        )

    stored_exp = bits_to_int(value.exponent)
    return FiniteParts(
        value.sign,
        (1 << p) | sig_bits,
        stored_exp - value.fmt.bias - p,
    )


def pack_rational(
    sign: int,
    numerator: int,
    denominator: int,
    fmt: FloatFormat,
    rounding: RoundingMode,
) -> tuple[object, Set[FloatingPointFlag]]:
    """Round a positive rational magnitude into an IEEEFloat."""
    if sign not in (0, 1):
        raise ValueError(f"sign must be 0 or 1, got {sign!r}")
    if numerator < 0 or denominator <= 0:
        raise ValueError("Expected a non-negative numerator and positive denominator.")

    flags: Set[FloatingPointFlag] = set()
    p = fmt.significand_bits

    if numerator == 0:
        return (_make_zero(sign, fmt), flags)

    actual_exp = _floor_log2_ratio(numerator, denominator)

    if actual_exp > fmt.max_actual_exponent:
        return _overflow_result(sign, fmt, rounding)

    if actual_exp >= fmt.min_actual_exponent_normal:
        scale = p - actual_exp
        rounded, inexact = _scaled_div_round(
            numerator, denominator, scale, rounding, sign
        )
        if inexact:
            flags.add(FloatingPointFlag.INEXACT)

        if rounded >= (1 << (p + 1)):
            rounded >>= 1
            actual_exp += 1

        if actual_exp > fmt.max_actual_exponent:
            overflow_value, overflow_flags = _overflow_result(sign, fmt, rounding)
            return overflow_value, flags | overflow_flags

        stored_exp = actual_exp + fmt.bias
        sig = rounded - (1 << p)
        return (
            _instantiate(
                sign,
                int_to_bits(stored_exp, fmt.exponent_bits),
                int_to_bits(sig, p),
                fmt,
            ),
            flags,
        )

    scale = p - fmt.min_actual_exponent_normal
    rounded, inexact = _scaled_div_round(numerator, denominator, scale, rounding, sign)
    if inexact:
        flags.add(FloatingPointFlag.INEXACT)
        flags.add(FloatingPointFlag.UNDERFLOW)

    if rounded == 0:
        return (
            _instantiate(sign, [0] * fmt.exponent_bits, [0] * fmt.significand_bits, fmt),
            flags,
        )

    if rounded >= (1 << p):
        return (
            _instantiate(sign, int_to_bits(1, fmt.exponent_bits), [0] * p, fmt),
            flags,
        )

    return (
        _instantiate(sign, [0] * fmt.exponent_bits, int_to_bits(rounded, p), fmt),
        flags,
    )


def pack_scaled_integer(
    sign: int,
    significand: int,
    exponent: int,
    fmt: FloatFormat,
    rounding: RoundingMode,
) -> tuple[object, Set[FloatingPointFlag]]:
    """Round significand * 2**exponent into an IEEEFloat."""
    if significand < 0:
        raise ValueError("significand must be non-negative.")
    if exponent >= 0:
        return pack_rational(sign, significand << exponent, 1, fmt, rounding)
    return pack_rational(sign, significand, 1 << (-exponent), fmt, rounding)


def _floor_log2_ratio(numerator: int, denominator: int) -> int:
    """Return floor(log2(numerator / denominator))."""
    exponent = numerator.bit_length() - denominator.bit_length()
    if not _ratio_ge_power_of_two(numerator, denominator, exponent):
        exponent -= 1
    while _ratio_ge_power_of_two(numerator, denominator, exponent + 1):
        exponent += 1
    return exponent


def _ratio_ge_power_of_two(numerator: int, denominator: int, exponent: int) -> bool:
    if exponent >= 0:
        return numerator >= (denominator << exponent)
    return (numerator << (-exponent)) >= denominator


def _scaled_div_round(
    numerator: int,
    denominator: int,
    scale: int,
    rounding: RoundingMode,
    sign: int,
) -> tuple[int, bool]:
    """Round numerator * 2**scale / denominator to an integer."""
    if scale >= 0:
        q, rem = divmod(numerator << scale, denominator)
    else:
        q, rem = divmod(numerator, denominator << (-scale))

    if rem == 0:
        return q, False

    round_up = False
    if rounding == RoundingMode.ROUND_TOWARD_POS_INF:
        round_up = sign == 0
    elif rounding == RoundingMode.ROUND_TOWARD_NEG_INF:
        round_up = sign == 1
    elif rounding == RoundingMode.ROUND_TO_NEAREST_EVEN:
        twice_rem = rem * 2
        cmp_den = denominator if scale >= 0 else denominator << (-scale)
        if twice_rem > cmp_den:
            round_up = True
        elif twice_rem == cmp_den:
            round_up = q % 2 == 1

    return q + 1 if round_up else q, True


def _overflow_result(
    sign: int,
    fmt: FloatFormat,
    rounding: RoundingMode,
) -> tuple[object, Set[FloatingPointFlag]]:
    flags = {FloatingPointFlag.OVERFLOW, FloatingPointFlag.INEXACT}
    infinity = (
        rounding == RoundingMode.ROUND_TO_NEAREST_EVEN
        or (rounding == RoundingMode.ROUND_TOWARD_POS_INF and sign == 0)
        or (rounding == RoundingMode.ROUND_TOWARD_NEG_INF and sign == 1)
    )
    if infinity:
        return (_make_inf(sign, fmt), flags)

    return (
        _instantiate(
            sign,
            int_to_bits(fmt.max_stored_exponent - 1, fmt.exponent_bits),
            [1] * fmt.significand_bits,
            fmt,
        ),
        flags,
    )
