"""Classification of IEEE 754 special values.

IEEE 754 reserves certain bit patterns for values that aren't "normal" numbers:
zero, infinity, NaN, and subnormals.  The encoding rules are surprisingly elegant
once you stare at them long enough (results may vary).
"""

from __future__ import annotations

import enum
from typing import List


class SpecialKind(enum.Enum):
    """All possible kinds of IEEE 754 values."""

    NORMAL = "normal"
    SUBNORMAL = "subnormal"
    POSITIVE_ZERO = "positive_zero"
    NEGATIVE_ZERO = "negative_zero"
    POSITIVE_INF = "positive_inf"
    NEGATIVE_INF = "negative_inf"
    NAN = "nan"


def classify(sign: int, exponent: List[int], mantissa: List[int]) -> SpecialKind:
    """Classify an IEEE 754 value from its raw bit fields.

    Args:
        sign: The sign bit (0 = positive, 1 = negative).
        exponent: Big-endian list of exponent bits.
        mantissa: Big-endian list of mantissa bits.

    Returns:
        The SpecialKind for this combination of bits.

    The rules, straight from the IEEE 754 spec (paraphrased for sanity):
      - exp = all ones,  mant = all zeros  →  ±Inf
      - exp = all ones,  mant ≠ all zeros  →  NaN  (sign is ignored for NaN)
      - exp = all zeros, mant = all zeros  →  ±Zero
      - exp = all zeros, mant ≠ all zeros  →  Subnormal
      - anything else                      →  Normal
    """
    exp_all_ones = all(b == 1 for b in exponent)
    exp_all_zeros = all(b == 0 for b in exponent)
    mant_all_zeros = all(b == 0 for b in mantissa)

    if exp_all_ones:
        if mant_all_zeros:
            return SpecialKind.POSITIVE_INF if sign == 0 else SpecialKind.NEGATIVE_INF
        return SpecialKind.NAN

    if exp_all_zeros:
        if mant_all_zeros:
            return SpecialKind.POSITIVE_ZERO if sign == 0 else SpecialKind.NEGATIVE_ZERO
        return SpecialKind.SUBNORMAL

    return SpecialKind.NORMAL
