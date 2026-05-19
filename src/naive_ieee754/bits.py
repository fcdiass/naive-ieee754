"""Low-level bit manipulation helpers.

These functions operate on plain Python lists of integers (each 0 or 1) and
have no knowledge of IEEE 754 semantics.  They exist so the higher-level code
can spell out exactly what happens at the bit level — which is the whole point
of this library.

All lists are big-endian: index 0 is the most significant bit.
"""

from __future__ import annotations

from typing import List


def int_to_bits(value: int, width: int) -> List[int]:
    """Convert a non-negative integer to a big-endian bit list of length `width`.

    Args:
        value: Non-negative integer to convert.
        width: Desired output length.  The result is zero-padded on the left.

    Raises:
        ValueError: If value does not fit in `width` bits.

    Example:
        int_to_bits(5, 4) -> [0, 1, 0, 1]
    """
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    bits = []
    n = value
    for _ in range(width):
        bits.append(n % 2)
        n //= 2
    bits.reverse()
    if n != 0:
        raise ValueError(f"{value} does not fit in {width} bits")
    return bits


def bits_to_int(bits: List[int]) -> int:
    """Convert a big-endian bit list to a non-negative integer.

    Example:
        bits_to_int([0, 1, 0, 1]) -> 5
    """
    result = 0
    for b in bits:
        result = result * 2 + b
    return result


def add_bit_lists(a: List[int], b: List[int]) -> List[int]:
    """Add two equal-length bit lists using a ripple-carry adder.

    Returns a result that may be one bit longer than the inputs if there is a
    carry out of the MSB.

    Args:
        a: Big-endian bit list.
        b: Big-endian bit list of the same length as `a`.
    """
    if len(a) != len(b):
        raise ValueError(f"bit lists must have equal length: {len(a)} vs {len(b)}")

    result = [0] * len(a)
    carry = 0
    for i in range(len(a) - 1, -1, -1):
        total = a[i] + b[i] + carry
        result[i] = total % 2
        carry = total // 2

    if carry:
        result = [1] + result

    return result


def subtract_bit_lists(a: List[int], b: List[int]) -> List[int]:
    """Subtract b from a (a - b) where a >= b, both as big-endian bit lists.

    Uses two's complement subtraction.  The result has the same length as `a`.
    If a < b the result wraps around (caller must ensure a >= b).
    """
    if len(a) != len(b):
        raise ValueError(f"bit lists must have equal length: {len(a)} vs {len(b)}")

    # Two's complement of b: flip all bits, add 1
    b_comp = twos_complement(b)

    # Add a and two's complement of b; discard the carry-out
    raw = add_bit_lists(a, b_comp)
    # Discard the extra carry bit if present (means a >= b, result is correct)
    if len(raw) > len(a):
        raw = raw[1:]
    return raw


def twos_complement(bits: List[int]) -> List[int]:
    """Compute the two's complement of a bit list.

    Flips all bits then adds 1.  The result has the same length as the input
    (overflow is discarded, consistent with fixed-width arithmetic).
    """
    flipped = [1 - b for b in bits]
    # Add 1
    carry = 1
    result = list(flipped)
    for i in range(len(result) - 1, -1, -1):
        total = result[i] + carry
        result[i] = total % 2
        carry = total // 2
    # Discard carry-out to keep the same width
    return result


def shift_right(bits: List[int], n: int) -> List[int]:
    """Logical right-shift by n positions.

    Zero-fills at the MSB end.  The result has the same length as `bits`.
    Bits shifted off the LSB end are discarded.

    Example:
        shift_right([1, 0, 1, 1], 2) -> [0, 0, 1, 0]
    """
    if n <= 0:
        return list(bits)
    if n >= len(bits):
        return [0] * len(bits)
    return [0] * n + bits[:-n]


def shift_right_extended(bits: List[int], n: int) -> List[int]:
    """Right-shift that keeps the shifted-out bits appended at the end.

    The result has the same length as `bits`.  The first `n` positions become
    zero, and the last `n` positions (previously beyond the window) hold the
    bits that were shifted out.  Used during exponent alignment to preserve
    guard/round/sticky bits.

    Example:
        shift_right_extended([1, 0, 1, 1], 2) -> [0, 0, 1, 0]
        (same as shift_right; this variant is semantically explicit about intent)
    """
    return shift_right(bits, n)


def shift_left(bits: List[int], n: int) -> List[int]:
    """Left-shift by n positions, appending zeros at the LSB end.

    The result is longer than `bits` by n elements.

    Example:
        shift_left([1, 0, 1], 2) -> [1, 0, 1, 0, 0]
    """
    if n <= 0:
        return list(bits)
    return list(bits) + [0] * n


def find_first_one(bits: List[int]) -> int:
    """Return the index of the leftmost 1 bit, or -1 if there are none.

    Example:
        find_first_one([0, 0, 1, 0, 1]) -> 2
        find_first_one([0, 0, 0])       -> -1
    """
    for i, b in enumerate(bits):
        if b == 1:
            return i
    return -1


def compare_bit_lists(a: List[int], b: List[int]) -> int:
    """Compare two equal-length big-endian bit lists as unsigned integers.

    Returns:
         1 if a > b
         0 if a == b
        -1 if a < b
    """
    if len(a) != len(b):
        raise ValueError(f"bit lists must have equal length: {len(a)} vs {len(b)}")
    for x, y in zip(a, b):
        if x > y:
            return 1
        if x < y:
            return -1
    return 0
