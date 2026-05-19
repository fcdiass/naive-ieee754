"""IEEE 754 rounding modes.

When the exact result of an operation has more precision than the format can
store, we have to round.  IEEE 754 defines four modes; the default
(round-to-nearest-even) is why 0.5 rounds to 0 and 1.5 rounds to 2 — it
minimizes accumulated rounding bias over many operations.
"""

from __future__ import annotations

import enum
from typing import List, Tuple


class RoundingMode(enum.Enum):
    """IEEE 754 rounding modes.

    The default is ROUND_TO_NEAREST_EVEN (also called "banker's rounding").
    The others are useful for interval arithmetic or just for watching things
    go wrong in different directions.
    """

    ROUND_TO_NEAREST_EVEN = "rne"
    """Round to the nearest representable value; ties go to the even one."""

    ROUND_TOWARD_ZERO = "rtz"
    """Truncate toward zero.  Always loses magnitude, never gains it."""

    ROUND_TOWARD_POS_INF = "rpi"
    """Always round up (toward positive infinity).  Ceiling function vibes."""

    ROUND_TOWARD_NEG_INF = "rni"
    """Always round down (toward negative infinity).  Floor function vibes."""


def apply_rounding(
    bits: List[int],
    keep: int,
    mode: RoundingMode,
    sign: int,
) -> Tuple[List[int], bool]:
    """Trim a bit list to `keep` bits, applying the given rounding mode.

    This function implements the guard/round/sticky rounding logic described in
    the IEEE 754 standard, expressed as explicit bit operations so you can see
    exactly what happens.

    Args:
        bits: Big-endian list of mantissa bits (without the implicit leading 1).
              Length must be >= keep.
        keep: How many bits to retain.
        mode: The rounding mode to apply.
        sign: The sign of the number being rounded (0 = positive, 1 = negative).

    Returns:
        A tuple of:
          - The rounded mantissa as a list of `keep` bits.  May be one bit
            longer if rounding caused a carry-out (caller must handle this).
          - precision_lost: True if any bits were discarded (regardless of
            whether the value actually changed).

    The three extra bits used for rounding decisions:
      - guard:  bits[keep]          (first discarded bit)
      - round:  bits[keep + 1]      (second discarded bit, if present)
      - sticky: OR of bits[keep+2:] (any remaining discarded bits)

    Round-to-nearest-even decision table:
      guard=0              → truncate (round down in magnitude)
      guard=1, sticky=1    → round up
      guard=1, sticky=0    → tie: round to even (up if last kept bit is 1,
                             down if last kept bit is 0)
    """
    if len(bits) <= keep:
        padding = [0] * (keep - len(bits))
        return list(bits) + padding, False

    kept = list(bits[:keep])
    discarded = bits[keep:]
    precision_lost = any(b == 1 for b in discarded)

    guard = discarded[0] if len(discarded) > 0 else 0
    round_bit = discarded[1] if len(discarded) > 1 else 0
    sticky = any(b == 1 for b in discarded[2:]) if len(discarded) > 2 else 0

    round_up = False

    if mode == RoundingMode.ROUND_TOWARD_ZERO:
        round_up = False

    elif mode == RoundingMode.ROUND_TOWARD_POS_INF:
        round_up = (sign == 0) and precision_lost

    elif mode == RoundingMode.ROUND_TOWARD_NEG_INF:
        round_up = (sign == 1) and precision_lost

    elif mode == RoundingMode.ROUND_TO_NEAREST_EVEN:
        if guard == 0:
            round_up = False
        elif sticky or round_bit:
            round_up = True
        else:
            # Exact tie: round to even (last kept bit determines direction)
            last_kept = kept[-1] if kept else 0
            round_up = last_kept == 1

    if not round_up:
        return kept, precision_lost

    # Add 1 to the kept bits (ripple carry from LSB)
    carry = 1
    result = list(kept)
    for i in range(len(result) - 1, -1, -1):
        total = result[i] + carry
        result[i] = total % 2
        carry = total // 2

    if carry:
        # Carry propagated beyond the MSB — the caller must increment the exponent
        result = [1] + result

    return result, precision_lost
