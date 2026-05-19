"""The IEEEFloat class, the heart of naive-ieee754."""

from __future__ import annotations

import math
from abc import ABC
from typing import TYPE_CHECKING, ClassVar, List

from ._binary import _FMT_TO_CLASS, _instantiate, pack_rational
from .bits import bits_to_int, int_to_bits
from .formats import (
    FloatFormat,
)
from .rounding import RoundingMode
from .special import SpecialKind, classify

if TYPE_CHECKING:
    from .arithmetic import ArithmeticResult


class IEEEFloat(ABC):
    """Abstract base for every IEEE 754 floating-point number.

    A floating-point number always belongs to a specific format (Float8,
    Float16, Float32, Float64, or a custom one).  This class is *not*
    instantiable on its own — concrete subclasses declare their format
    via a ``FORMAT`` class attribute and inherit all the behaviour below.

    Example::

        class Float32(IEEEFloat):
            FORMAT = FLOAT32_FORMAT

    Attributes (per instance):
        sign: 0 for positive, 1 for negative.
        exponent: Big-endian exponent bits.
        significand: Big-endian significand (fractional) bits.
        fmt: The format this number belongs to (== ``type(self).FORMAT``).

    Instances are immutable after construction.
    """

    __slots__ = ("sign", "exponent", "significand", "fmt")

    # Every concrete subclass must override this with a FloatFormat instance.
    FORMAT: ClassVar[FloatFormat]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate the subclass and auto-register its FORMAT."""
        super().__init_subclass__(**kwargs)
        fmt = cls.__dict__.get("FORMAT")
        if not isinstance(fmt, FloatFormat):
            raise TypeError(
                f"{cls.__name__} must define a FORMAT class attribute "
                f"(an instance of FloatFormat)."
            )
        _FMT_TO_CLASS[fmt] = cls

    def __init__(
        self,
        sign: int,
        exponent: List[int],
        significand: List[int],
    ) -> None:
        cls = type(self)
        if cls is IEEEFloat:
            raise TypeError(
                "IEEEFloat is abstract — instantiate a concrete subclass "
                "such as Float8, Float16, Float32, Float64, or one created "
                "via custom_float()."
            )
        fmt = cls.FORMAT
        if sign not in (0, 1):
            raise ValueError(f"sign must be 0 or 1, got {sign!r}")
        if len(exponent) != fmt.exponent_bits:
            raise ValueError(
                f"exponent must have {fmt.exponent_bits} bits, got {len(exponent)}"
            )
        if len(significand) != fmt.significand_bits:
            raise ValueError(
                f"significand must have {fmt.significand_bits} bits, got {len(significand)}"
            )
        object.__setattr__(self, "sign", sign)
        object.__setattr__(self, "exponent", list(exponent))
        object.__setattr__(self, "significand", list(significand))
        object.__setattr__(self, "fmt", fmt)

    def __setattr__(self, name: str, value: object) -> None:  # type: ignore[override]
        raise AttributeError("IEEEFloat is immutable")

    @property
    def kind(self) -> SpecialKind:
        """Classify this value according to IEEE 754 encoding rules."""
        return classify(self.sign, self.exponent, self.significand)

    def is_nan(self) -> bool:
        return self.kind == SpecialKind.NAN

    def is_inf(self) -> bool:
        return self.kind in (SpecialKind.POSITIVE_INF, SpecialKind.NEGATIVE_INF)

    def is_zero(self) -> bool:
        return self.kind in (SpecialKind.POSITIVE_ZERO, SpecialKind.NEGATIVE_ZERO)

    def is_subnormal(self) -> bool:
        return self.kind == SpecialKind.SUBNORMAL

    def is_normal(self) -> bool:
        return self.kind == SpecialKind.NORMAL

    def is_positive(self) -> bool:
        """True for positive numbers, +0, +Inf, and positive-sign NaNs."""
        return self.sign == 0

    def is_negative(self) -> bool:
        """True for negative numbers, -0, -Inf, and negative-sign NaNs."""
        return self.sign == 1

    @classmethod
    def positive_zero(cls) -> IEEEFloat:
        """Return +0 in this format."""
        fmt = cls.FORMAT
        return _instantiate(0, [0] * fmt.exponent_bits, [0] * fmt.significand_bits, fmt)

    @classmethod
    def negative_zero(cls) -> IEEEFloat:
        """Return -0 in this format."""
        fmt = cls.FORMAT
        return _instantiate(1, [0] * fmt.exponent_bits, [0] * fmt.significand_bits, fmt)

    @classmethod
    def positive_infinity(cls) -> IEEEFloat:
        """Return +Inf in this format."""
        fmt = cls.FORMAT
        return _instantiate(0, [1] * fmt.exponent_bits, [0] * fmt.significand_bits, fmt)

    @classmethod
    def negative_infinity(cls) -> IEEEFloat:
        """Return -Inf in this format."""
        fmt = cls.FORMAT
        return _instantiate(1, [1] * fmt.exponent_bits, [0] * fmt.significand_bits, fmt)

    @classmethod
    def nan(cls) -> IEEEFloat:
        """Return a canonical quiet NaN in this format."""
        fmt = cls.FORMAT
        significand = [0] * fmt.significand_bits
        if fmt.significand_bits > 0:
            significand[0] = 1
        return _instantiate(0, [1] * fmt.exponent_bits, significand, fmt)

    @classmethod
    def from_decimal(
        cls,
        value: float,
        rounding: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
    ) -> IEEEFloat:
        """Convert a Python float or int to this format."""
        fmt = cls.FORMAT
        if isinstance(value, int) and not isinstance(value, bool):
            if value == 0:
                return cls.positive_zero()
            sign = 1 if value < 0 else 0
            result, _ = pack_rational(sign, abs(value), 1, fmt, rounding)
            return result

        value = float(value)

        if math.isnan(value):
            return cls.nan()
        if math.isinf(value):
            return cls.positive_infinity() if value > 0 else cls.negative_infinity()
        if value == 0.0:
            return (
                cls.negative_zero()
                if math.copysign(1.0, value) < 0
                else cls.positive_zero()
            )

        sign = 1 if value < 0 else 0
        numerator, denominator = abs(value).as_integer_ratio()
        result, _ = pack_rational(sign, numerator, denominator, fmt, rounding)
        return result

    @classmethod
    def from_bits(cls, bits: List[int]) -> IEEEFloat:
        """Construct an instance from [sign | exponent | significand] bits."""
        fmt = cls.FORMAT
        if len(bits) != fmt.total_bits:
            raise ValueError(
                f"Expected {fmt.total_bits} bits for {fmt.name}, got {len(bits)}"
            )
        sign = bits[0]
        exponent = list(bits[1 : 1 + fmt.exponent_bits])
        significand = list(bits[1 + fmt.exponent_bits :])
        return _instantiate(sign, exponent, significand, fmt)

    @classmethod
    def from_raw_int(cls, pattern: int) -> IEEEFloat:
        """Construct an instance from a raw integer bit pattern."""
        bits = int_to_bits(pattern, cls.FORMAT.total_bits)
        return cls.from_bits(bits)

    @classmethod
    def from_bit_string(cls, s: str) -> IEEEFloat:
        """Construct an instance from a bit string like '0 10000000 10010010000111010011001'.

        Spaces are ignored, so both '0 10000000 ...' and '010000000...' are accepted.
        """
        bits = [int(c) for c in s if c in "01"]
        return cls.from_bits(bits)

    def to_decimal(self) -> float:
        """Reconstruct the Python float value corresponding to this bit pattern."""
        kind = self.kind

        if kind == SpecialKind.NAN:
            return float("nan")
        if kind == SpecialKind.POSITIVE_INF:
            return float("inf")
        if kind == SpecialKind.NEGATIVE_INF:
            return float("-inf")
        if kind in (SpecialKind.POSITIVE_ZERO, SpecialKind.NEGATIVE_ZERO):
            return math.copysign(0.0, -1.0 if self.sign else 1.0)

        p = self.fmt.significand_bits
        sig_bits = bits_to_int(self.significand)
        if kind == SpecialKind.SUBNORMAL:
            significand = sig_bits
            exponent = self.fmt.min_actual_exponent_normal - p
        else:
            significand = (1 << p) | sig_bits
            stored_exp = bits_to_int(self.exponent)
            exponent = stored_exp - self.fmt.bias - p

        value = math.ldexp(float(significand), exponent)
        return -value if self.sign else value

    def to_bit_string(self) -> str:
        """Return the bit pattern as 'sign exponent significand'."""
        sign_str = str(self.sign)
        exp_str = "".join(str(b) for b in self.exponent)
        sig_str = "".join(str(b) for b in self.significand)
        return f"{sign_str} {exp_str} {sig_str}"

    def to_hex(self) -> str:
        """Return the full bit pattern as a zero-padded hexadecimal string."""
        all_bits = [self.sign] + self.exponent + self.significand
        value = bits_to_int(all_bits)
        hex_digits = math.ceil(self.fmt.total_bits / 4)
        return format(value, f"0{hex_digits}X")

    def to_fields_dict(self) -> dict:
        """Return the bit fields and their interpreted values."""
        kind = self.kind
        stored_exp = bits_to_int(self.exponent)
        if kind == SpecialKind.NORMAL:
            actual_exp = stored_exp - self.fmt.bias
        elif kind == SpecialKind.SUBNORMAL:
            actual_exp = self.fmt.min_actual_exponent_normal
        else:
            actual_exp = None
        return {
            "sign": self.sign,
            "exponent_bits": list(self.exponent),
            "exponent_stored": stored_exp,
            "exponent_actual": actual_exp,
            "significand_bits": list(self.significand),
            "kind": kind.value,
            "decimal": self.to_decimal(),
        }

    def __repr__(self) -> str:
        exp_str = "".join(str(b) for b in self.exponent)
        sig_str = "".join(str(b) for b in self.significand)
        return f"{self.fmt.name}(sign={self.sign} exp=[{exp_str}] sig=[{sig_str}])"

    def __str__(self) -> str:
        kind = self.kind
        if kind == SpecialKind.NAN:
            return f"NaN ({self.fmt.name})"
        if kind == SpecialKind.POSITIVE_INF:
            return f"+Inf ({self.fmt.name})"
        if kind == SpecialKind.NEGATIVE_INF:
            return f"-Inf ({self.fmt.name})"
        if kind in (SpecialKind.POSITIVE_ZERO, SpecialKind.NEGATIVE_ZERO):
            prefix = "-" if self.sign else "+"
            return f"{prefix}0 ({self.fmt.name})"
        return f"{self.to_decimal()} ({self.fmt.name})"

    def __eq__(self, other: object) -> bool:
        """IEEE 754 equality: NaN != NaN, +0 == -0."""
        if not isinstance(other, IEEEFloat):
            return NotImplemented
        if self.is_nan() or other.is_nan():
            return False
        if self.is_zero() and other.is_zero():
            return True
        return (
            self.sign == other.sign
            and self.exponent == other.exponent
            and self.significand == other.significand
            and self.fmt == other.fmt
        )

    def __lt__(self, other: IEEEFloat) -> bool:
        if self.is_nan() or other.is_nan():
            return False
        return self.to_decimal() < other.to_decimal()

    def __le__(self, other: IEEEFloat) -> bool:
        if self.is_nan() or other.is_nan():
            return False
        return self.to_decimal() <= other.to_decimal()

    def __gt__(self, other: IEEEFloat) -> bool:
        if self.is_nan() or other.is_nan():
            return False
        return self.to_decimal() > other.to_decimal()

    def __ge__(self, other: IEEEFloat) -> bool:
        if self.is_nan() or other.is_nan():
            return False
        return self.to_decimal() >= other.to_decimal()

    def __hash__(self) -> int:
        return hash((self.sign, tuple(self.exponent), tuple(self.significand), self.fmt))

    def __neg__(self) -> IEEEFloat:
        """Flip the sign bit. Works for all values including NaN."""
        return _instantiate(1 - self.sign, self.exponent, self.significand, self.fmt)

    def __abs__(self) -> IEEEFloat:
        """Clear the sign bit."""
        return _instantiate(0, self.exponent, self.significand, self.fmt)

    def __add__(self, other: IEEEFloat) -> IEEEFloat:
        _check_same_format(self, other)
        from .arithmetic import _add_impl

        return _add_impl(self, other, RoundingMode.ROUND_TO_NEAREST_EVEN).result

    def __sub__(self, other: IEEEFloat) -> IEEEFloat:
        _check_same_format(self, other)
        from .arithmetic import _add_impl

        return _add_impl(
            self, other, RoundingMode.ROUND_TO_NEAREST_EVEN, _negate_b=True
        ).result

    def __mul__(self, other: IEEEFloat) -> IEEEFloat:
        _check_same_format(self, other)
        from .arithmetic import _mul_impl

        return _mul_impl(self, other, RoundingMode.ROUND_TO_NEAREST_EVEN).result

    def __truediv__(self, other: IEEEFloat) -> IEEEFloat:
        _check_same_format(self, other)
        from .arithmetic import _div_impl

        return _div_impl(self, other, RoundingMode.ROUND_TO_NEAREST_EVEN).result

    def add(
        self,
        other: IEEEFloat,
        rounding: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
    ) -> ArithmeticResult:
        """Add two IEEEFloat values, returning a step-by-step ArithmeticResult."""
        _check_same_format(self, other)
        from .arithmetic import _add_impl

        return _add_impl(self, other, rounding)

    def sub(
        self,
        other: IEEEFloat,
        rounding: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
    ) -> ArithmeticResult:
        """Subtract other from self, returning a step-by-step ArithmeticResult."""
        _check_same_format(self, other)
        from .arithmetic import _add_impl

        return _add_impl(self, other, rounding, _negate_b=True)

    def mul(
        self,
        other: IEEEFloat,
        rounding: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
    ) -> ArithmeticResult:
        """Multiply two IEEEFloat values, returning a step-by-step ArithmeticResult."""
        _check_same_format(self, other)
        from .arithmetic import _mul_impl

        return _mul_impl(self, other, rounding)

    def div(
        self,
        other: IEEEFloat,
        rounding: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
    ) -> ArithmeticResult:
        """Divide self by other, returning a step-by-step ArithmeticResult."""
        _check_same_format(self, other)
        from .arithmetic import _div_impl

        return _div_impl(self, other, rounding)

    def sqrt(
        self,
        rounding: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
    ) -> ArithmeticResult:
        """Compute the correctly-rounded square root, returning a step-by-step ArithmeticResult."""
        from .arithmetic import _sqrt_impl

        return _sqrt_impl(self, rounding)


def _check_same_format(a: IEEEFloat, b: IEEEFloat) -> None:
    if a.fmt != b.fmt:
        raise ValueError(
            f"Cannot mix formats: {a.fmt.name} and {b.fmt.name}. "
            f"Convert both operands to the same format first."
        )
