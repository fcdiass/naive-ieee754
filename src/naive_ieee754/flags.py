"""IEEE 754 floating-point status flags."""

from __future__ import annotations

import enum


class FloatingPointFlag(enum.Enum):
    """IEEE 754 exception/status flags raised by arithmetic operations."""

    INVALID = "invalid"
    DIVIDE_BY_ZERO = "divide_by_zero"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    INEXACT = "inexact"
