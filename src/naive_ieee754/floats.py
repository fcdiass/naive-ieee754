"""Concrete IEEE 754 float types and the custom_float() factory."""

from __future__ import annotations

from .core import IEEEFloat
from .formats import (
    FLOAT8_FORMAT,
    FLOAT16_FORMAT,
    FLOAT32_FORMAT,
    FLOAT64_FORMAT,
    FloatFormat,
)


class Float8(IEEEFloat):
    """8-bit minifloat (1 sign + 4 exponent + 3 significand).

    Fits in a single byte.  Represents about 238 distinct finite values.
    Great for seeing IEEE 754 quirks without needing a microscope.
    """

    FORMAT = FLOAT8_FORMAT


class Float16(IEEEFloat):
    """IEEE 754 half-precision (1 sign + 5 exponent + 10 significand)."""

    FORMAT = FLOAT16_FORMAT


class Float32(IEEEFloat):
    """IEEE 754 single-precision (1 sign + 8 exponent + 23 significand).

    The classic C ``float``.  This is the one that makes ``0.1 + 0.2``
    not equal ``0.3``.  Don't @ us, that's on IEEE 754.
    """

    FORMAT = FLOAT32_FORMAT


class Float64(IEEEFloat):
    """IEEE 754 double-precision (1 sign + 11 exponent + 52 significand).

    Python's native ``float`` is a Float64 in disguise.  Still not enough
    precision to represent ``0.1`` exactly — but at least it fails with
    more decimal places.
    """

    FORMAT = FLOAT64_FORMAT


def custom_float(exponent_bits: int, significand_bits: int) -> type:
    """Return a new IEEEFloat subclass for a custom IEEE 754-like format.

    Unlike the predefined Float8/Float16/Float32/Float64, this returns a
    class that you assign to a name and use exactly like the predefined ones.

    Example::

        Mini = custom_float(exponent_bits=4, significand_bits=3)
        x = Mini.from_decimal(1.5)
        y = Mini(0, [0, 1, 1, 1], [1, 0, 0])  # direct construction
        number_line(Mini.FORMAT)
    """
    if exponent_bits < 2:
        raise ValueError("exponent_bits must be >= 2")
    if significand_bits < 1:
        raise ValueError("significand_bits must be >= 1")
    fmt = FloatFormat(
        name=f"Custom({exponent_bits}e+{significand_bits}sig)",
        exponent_bits=exponent_bits,
        significand_bits=significand_bits,
    )
    # IEEEFloat.__init_subclass__ auto-registers the new class in _FMT_TO_CLASS.
    return type(fmt.name, (IEEEFloat,), {"FORMAT": fmt})
