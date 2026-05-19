"""IEEE 754 format definitions.

Each format is described by how many bits it dedicates to the exponent and
significand fields.  The sign field is always 1 bit — IEEE 754 is nothing if not
consistent about that one thing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloatFormat:
    """Describes an IEEE 754 floating-point format.

    Attributes:
        name: Human-readable name (e.g. "Float32").
        exponent_bits: Number of bits in the exponent field.
        significand_bits: Number of bits in the significand (fractional) field.
            IEEE 754-2019 uses "significand"; older literature says "mantissa".

    The sign field is always 1 bit.  Total bits = 1 + exponent_bits + significand_bits.
    """

    name: str
    exponent_bits: int
    significand_bits: int

    @property
    def total_bits(self) -> int:
        return 1 + self.exponent_bits + self.significand_bits

    @property
    def bias(self) -> int:
        """Exponent bias: the value subtracted from the stored exponent to get the actual exponent.

        For a format with E exponent bits: bias = 2^(E-1) - 1.
        Float32 bias is 127, Float64 bias is 1023.
        """
        return (2 ** (self.exponent_bits - 1)) - 1

    @property
    def max_stored_exponent(self) -> int:
        """Maximum value that fits in the exponent field (all ones).

        This value is reserved for Inf and NaN, so the largest usable stored
        exponent for normal numbers is max_stored_exponent - 1.
        """
        return (2**self.exponent_bits) - 1

    @property
    def max_actual_exponent(self) -> int:
        """Largest actual (unbiased) exponent for a normal number."""
        return self.max_stored_exponent - 1 - self.bias

    @property
    def min_actual_exponent_normal(self) -> int:
        """Smallest actual (unbiased) exponent for a normal number.

        Stored exponent 1 maps to actual exponent 1 - bias.
        Stored exponent 0 is reserved for zero and subnormals.
        """
        return 1 - self.bias

    @property
    def max_normal(self) -> float:
        """Largest representable finite value."""
        significand = 2.0 - 2.0 ** (-self.significand_bits)
        return significand * (2.0**self.max_actual_exponent)

    @property
    def min_positive_normal(self) -> float:
        """Smallest positive normal number (stored exponent = 1)."""
        return 2.0**self.min_actual_exponent_normal

    @property
    def eps(self) -> float:
        """Machine epsilon: smallest positive value such that 1 + eps > 1.

        Equal to one ULP of 1.0: 2^(-significand_bits).
        """
        return 2.0 ** (-self.significand_bits)

    @property
    def min_positive_subnormal(self) -> float:
        """Smallest positive subnormal number (only the LSB of the significand is set).

        This property is also used by the visualization module to shade the
        subnormal region.
        """
        return 2.0 ** (self.min_actual_exponent_normal - self.significand_bits)


# ---------------------------------------------------------------------------
# Predefined formats
# ---------------------------------------------------------------------------

FLOAT8_FORMAT = FloatFormat(
    name="Float8",
    exponent_bits=4,
    significand_bits=3,
)
"""8-bit minifloat: 1 sign + 4 exponent + 3 significand.

Fits in a single byte.  Represents about 238 distinct finite values.
Your GPS uses more precision to tell you where the nearest coffee shop is.
"""

FLOAT16_FORMAT = FloatFormat(
    name="Float16",
    exponent_bits=5,
    significand_bits=10,
)
"""IEEE 754 half-precision (binary16): 1 sign + 5 exponent + 10 significand."""

FLOAT32_FORMAT = FloatFormat(
    name="Float32",
    exponent_bits=8,
    significand_bits=23,
)
"""IEEE 754 single-precision (binary32): 1 sign + 8 exponent + 23 significand.

The classic "float" in C, Java, and most languages.  This is the one that
makes 0.1 + 0.2 = 0.30000001192092896.  Don't @ us, that's on IEEE 754.
"""

FLOAT64_FORMAT = FloatFormat(
    name="Float64",
    exponent_bits=11,
    significand_bits=52,
)
"""IEEE 754 double-precision (binary64): 1 sign + 11 exponent + 52 significand.

Python's native float is a Float64 in disguise.  Still not enough precision
to represent 0.1 exactly — but at least it fails with more decimal places.
"""
