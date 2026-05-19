"""Step-by-step IEEE 754 arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Set

from ._binary import (
    _instantiate,
    _make_inf,
    _make_nan,
    _make_zero,
    decode_finite,
    pack_rational,
    pack_scaled_integer,
)
from .flags import FloatingPointFlag
from .rounding import RoundingMode
from .special import SpecialKind

if TYPE_CHECKING:
    from .core import IEEEFloat


@dataclass
class Step:
    """One step in an arithmetic operation."""

    name: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"[{self.name}] {self.description}"]
        for k, v in self.details.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


@dataclass
class ArithmeticResult:
    """The result of a verbose arithmetic operation."""

    result: Any
    steps: List[Step]
    precision_lost: bool
    operation: str
    flags: Set[FloatingPointFlag] = field(default_factory=set)

    def explain(self) -> str:
        """Return a human-readable multi-line breakdown of all steps."""
        lines = [
            f"Operation: {self.operation}",
            f"Precision lost: {self.precision_lost}",
            f"Flags: {', '.join(flag.value for flag in sorted(self.flags, key=lambda f: f.value)) or 'none'}",
            "",
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step}")
            lines.append("")
        lines.append(f"Result: {self.result}")
        return "\n".join(lines)

    def precision_report(self) -> str:
        """Compare the rounded result with Python float arithmetic."""
        kind = self.result.kind
        if kind in (
            SpecialKind.NAN,
            SpecialKind.POSITIVE_INF,
            SpecialKind.NEGATIVE_INF,
        ):
            return f"Result is {kind.value}; precision report not applicable."

        represented = self.result.to_decimal()
        steps_input = next((s for s in self.steps if s.name == "input"), None)
        if steps_input is None:
            return "Could not find input step; precision report unavailable."

        a_val = steps_input.details.get("a_decimal", 0.0)
        b_val = steps_input.details.get("b_decimal", 0.0)
        op = self.operation

        if op == "add":
            exact = a_val + b_val
        elif op == "sub":
            exact = a_val - b_val
        elif op == "mul":
            exact = a_val * b_val
        elif op == "div":
            exact = a_val / b_val if b_val != 0 else float("inf")
        else:
            return "Unknown operation."

        abs_error = abs(represented - exact)
        rel_error = (abs_error / abs(exact)) if exact != 0 else 0.0

        lines = [
            f"  exact (Python float64): {exact!r}",
            f"  represented ({self.result.fmt.name}): {represented!r}",
            f"  absolute error: {abs_error!r}",
            f"  relative error: {rel_error:.2e}"
            if rel_error
            else "  relative error: 0",
        ]
        return "\n".join(lines)


def _add_impl(
    a: IEEEFloat,
    b: IEEEFloat,
    rounding: RoundingMode,
    _negate_b: bool = False,
) -> ArithmeticResult:
    """Implement IEEE 754 addition or subtraction."""
    fmt = a.fmt
    op_name = "sub" if _negate_b else "add"
    b_sign = 1 - b.sign if _negate_b else b.sign
    steps = [_input_step(a, b)]

    if a.is_nan() or b.is_nan():
        result = _make_nan(fmt)
        steps.append(
            Step(
                "special_check",
                "NaN operand produced a quiet NaN.",
                {"triggered": "nan"},
            )
        )
        return _finish(result, steps, set(), op_name)

    if a.is_inf() and b.is_inf():
        if a.sign == b_sign:
            result = (
                _make_inf(1, fmt)
                if a.sign
                else _make_inf(0, fmt)
            )
            steps.append(
                Step(
                    "special_check",
                    "Same-sign infinities produced the same infinity.",
                    {"triggered": "inf"},
                )
            )
            return _finish(result, steps, set(), op_name)
        result = _make_nan(fmt)
        steps.append(
            Step(
                "special_check",
                "Opposite infinities are an invalid operation.",
                {"triggered": "inf_minus_inf"},
            )
        )
        return _finish(result, steps, {FloatingPointFlag.INVALID}, op_name)

    if a.is_inf():
        result = _instantiate(a.sign, a.exponent, a.significand, fmt)
        steps.append(
            Step("special_check", "a is infinite; result is a.", {"triggered": "a_inf"})
        )
        return _finish(result, steps, set(), op_name)

    if b.is_inf():
        result = _instantiate(b_sign, b.exponent, b.significand, fmt)
        steps.append(
            Step(
                "special_check",
                "b is infinite; result is signed b.",
                {"triggered": "b_inf"},
            )
        )
        return _finish(result, steps, set(), op_name)

    if a.is_zero() and b.is_zero():
        result = _zero_sum(a.sign, b_sign, fmt, rounding)
        steps.append(
            Step("special_check", "Both operands are zero.", {"triggered": "zero_zero"})
        )
        return _finish(result, steps, set(), op_name)

    if a.is_zero():
        result = _instantiate(b_sign, b.exponent, b.significand, fmt)
        steps.append(
            Step(
                "special_check",
                "a is zero; result is b with its effective sign.",
                {"triggered": "a_zero"},
            )
        )
        return _finish(result, steps, set(), op_name)

    if b.is_zero():
        result = _instantiate(a.sign, a.exponent, a.significand, fmt)
        steps.append(
            Step("special_check", "b is zero; result is a.", {"triggered": "b_zero"})
        )
        return _finish(result, steps, set(), op_name)

    steps.append(
        Step(
            "special_check",
            "No special case; using exact integer addition.",
            {"triggered": "none"},
        )
    )

    pa = decode_finite(a)
    pb = decode_finite(b)
    common_exp = min(pa.exponent, pb.exponent)
    a_int = pa.significand << (pa.exponent - common_exp)
    b_int = pb.significand << (pb.exponent - common_exp)
    if pa.sign:
        a_int = -a_int
    if b_sign:
        b_int = -b_int

    exact = a_int + b_int
    steps.append(
        Step(
            "add_significands",
            "Aligned significands exactly and added signed integers.",
            {
                "common_exponent": common_exp,
                "a_integer": a_int,
                "b_integer": b_int,
                "sum_integer": exact,
            },
        )
    )

    if exact == 0:
        result = (
            _make_zero(1, fmt)
            if rounding == RoundingMode.ROUND_TOWARD_NEG_INF
            else _make_zero(0, fmt)
        )
        steps.append(
            Step(
                "normalize",
                "Exact cancellation produced signed zero.",
                {"result_sign": result.sign},
            )
        )
        return _finish(result, steps, set(), op_name)

    result_sign = 1 if exact < 0 else 0
    result, flags = pack_scaled_integer(
        result_sign, abs(exact), common_exp, fmt, rounding
    )
    steps.append(
        Step(
            "round",
            f"Rounded exact result with {rounding.value}.",
            {"flags": _flag_values(flags)},
        )
    )
    steps.append(_result_step(result))
    return _finish(result, steps, flags, op_name)


def _mul_impl(
    a: IEEEFloat,
    b: IEEEFloat,
    rounding: RoundingMode,
) -> ArithmeticResult:
    """Implement IEEE 754 multiplication."""
    fmt = a.fmt
    result_sign = a.sign ^ b.sign
    steps = [_input_step(a, b)]

    if a.is_nan() or b.is_nan():
        result = _make_nan(fmt)
        steps.append(
            Step(
                "special_check",
                "NaN operand produced a quiet NaN.",
                {"triggered": "nan"},
            )
        )
        return _finish(result, steps, set(), "mul")

    if a.is_inf() or b.is_inf():
        if a.is_zero() or b.is_zero():
            result = _make_nan(fmt)
            steps.append(
                Step(
                    "special_check",
                    "Zero times infinity is invalid.",
                    {"triggered": "zero_times_inf"},
                )
            )
            return _finish(result, steps, {FloatingPointFlag.INVALID}, "mul")
        result = (
            _make_inf(1, fmt)
            if result_sign
            else _make_inf(0, fmt)
        )
        steps.append(
            Step(
                "special_check",
                "Infinity times nonzero produced signed infinity.",
                {"triggered": "inf"},
            )
        )
        return _finish(result, steps, set(), "mul")

    if a.is_zero() or b.is_zero():
        result = (
            _make_zero(1, fmt)
            if result_sign
            else _make_zero(0, fmt)
        )
        steps.append(
            Step(
                "special_check",
                "Zero operand produced signed zero.",
                {"triggered": "zero"},
            )
        )
        return _finish(result, steps, set(), "mul")

    steps.append(
        Step(
            "special_check",
            "No special case; using exact integer multiplication.",
            {"triggered": "none"},
        )
    )
    steps.append(
        Step(
            "sign",
            "Result sign is the XOR of operand signs.",
            {"result_sign": result_sign},
        )
    )

    pa = decode_finite(a)
    pb = decode_finite(b)
    exponent = pa.exponent + pb.exponent
    product = pa.significand * pb.significand
    steps.append(
        Step(
            "exponent_add",
            "Added exact binary exponents.",
            {"result_exponent": exponent},
        )
    )
    steps.append(
        Step(
            "significand_multiply",
            "Multiplied full integer significands.",
            {"product": product},
        )
    )

    result, flags = pack_scaled_integer(result_sign, product, exponent, fmt, rounding)
    steps.append(
        Step(
            "round",
            f"Rounded exact product with {rounding.value}.",
            {"flags": _flag_values(flags)},
        )
    )
    steps.append(_result_step(result))
    return _finish(result, steps, flags, "mul")


def _div_impl(
    a: IEEEFloat,
    b: IEEEFloat,
    rounding: RoundingMode,
) -> ArithmeticResult:
    """Implement IEEE 754 division."""
    fmt = a.fmt
    result_sign = a.sign ^ b.sign
    steps = [_input_step(a, b)]

    if a.is_nan() or b.is_nan():
        result = _make_nan(fmt)
        steps.append(
            Step(
                "special_check",
                "NaN operand produced a quiet NaN.",
                {"triggered": "nan"},
            )
        )
        return _finish(result, steps, set(), "div")

    if a.is_zero() and b.is_zero():
        result = _make_nan(fmt)
        steps.append(
            Step(
                "special_check",
                "Zero divided by zero is invalid.",
                {"triggered": "zero_div_zero"},
            )
        )
        return _finish(result, steps, {FloatingPointFlag.INVALID}, "div")

    if a.is_inf() and b.is_inf():
        result = _make_nan(fmt)
        steps.append(
            Step(
                "special_check",
                "Infinity divided by infinity is invalid.",
                {"triggered": "inf_div_inf"},
            )
        )
        return _finish(result, steps, {FloatingPointFlag.INVALID}, "div")

    if b.is_zero():
        result = (
            _make_inf(1, fmt)
            if result_sign
            else _make_inf(0, fmt)
        )
        steps.append(
            Step(
                "special_check",
                "Finite nonzero divided by zero produced infinity.",
                {"triggered": "div_by_zero"},
            )
        )
        return _finish(result, steps, {FloatingPointFlag.DIVIDE_BY_ZERO}, "div")

    if a.is_zero():
        result = (
            _make_zero(1, fmt)
            if result_sign
            else _make_zero(0, fmt)
        )
        steps.append(
            Step(
                "special_check",
                "Zero numerator produced signed zero.",
                {"triggered": "zero_numerator"},
            )
        )
        return _finish(result, steps, set(), "div")

    if a.is_inf():
        result = (
            _make_inf(1, fmt)
            if result_sign
            else _make_inf(0, fmt)
        )
        steps.append(
            Step(
                "special_check",
                "Infinite numerator produced signed infinity.",
                {"triggered": "inf_numerator"},
            )
        )
        return _finish(result, steps, set(), "div")

    if b.is_inf():
        result = (
            _make_zero(1, fmt)
            if result_sign
            else _make_zero(0, fmt)
        )
        steps.append(
            Step(
                "special_check",
                "Finite divided by infinity produced signed zero.",
                {"triggered": "inf_denominator"},
            )
        )
        return _finish(result, steps, set(), "div")

    steps.append(
        Step(
            "special_check",
            "No special case; using exact rational division.",
            {"triggered": "none"},
        )
    )
    steps.append(
        Step(
            "sign",
            "Result sign is the XOR of operand signs.",
            {"result_sign": result_sign},
        )
    )

    pa = decode_finite(a)
    pb = decode_finite(b)
    exponent = pa.exponent - pb.exponent
    if exponent >= 0:
        numerator = pa.significand << exponent
        denominator = pb.significand
    else:
        numerator = pa.significand
        denominator = pb.significand << (-exponent)

    steps.append(
        Step(
            "exponent_subtract",
            "Subtracted exact binary exponents.",
            {"result_exponent": exponent},
        )
    )
    steps.append(
        Step(
            "significand_divide",
            "Built an exact rational quotient.",
            {"numerator": numerator, "denominator": denominator},
        )
    )

    result, flags = pack_rational(result_sign, numerator, denominator, fmt, rounding)
    steps.append(
        Step(
            "round",
            f"Rounded exact quotient with {rounding.value}.",
            {"flags": _flag_values(flags)},
        )
    )
    steps.append(_result_step(result))
    return _finish(result, steps, flags, "div")


def _sqrt_impl(
    a: IEEEFloat,
    rounding: RoundingMode,
) -> ArithmeticResult:
    """Implement IEEE 754 square root (correctly rounded)."""
    from math import isqrt

    fmt = a.fmt
    op_name = "sqrt"
    steps = [
        Step(
            "input",
            f"Input: a = {a.to_decimal()} ({a.to_bit_string()})",
            {"a_decimal": a.to_decimal(), "a_bits": a.to_bit_string()},
        )
    ]

    if a.is_nan():
        steps.append(
            Step(
                "special_check", "NaN input produces a quiet NaN.", {"triggered": "nan"}
            )
        )
        return _finish(_make_nan(fmt), steps, {FloatingPointFlag.INVALID}, op_name)

    if a.sign == 1 and not a.is_zero():
        steps.append(
            Step(
                "special_check",
                "Square root of a negative number is invalid.",
                {"triggered": "negative"},
            )
        )
        return _finish(_make_nan(fmt), steps, {FloatingPointFlag.INVALID}, op_name)

    if a.is_inf():
        steps.append(
            Step("special_check", "sqrt(+Inf) = +Inf.", {"triggered": "positive_inf"})
        )
        return _finish(_make_inf(0, fmt), steps, set(), op_name)

    if a.is_zero():
        result = _make_zero(a.sign, fmt)
        steps.append(
            Step(
                "special_check",
                "sqrt(±0) = ±0 with sign preserved.",
                {"triggered": "zero"},
            )
        )
        return _finish(result, steps, set(), op_name)

    steps.append(
        Step(
            "special_check",
            "No special case; computing integer square root.",
            {"triggered": "none"},
        )
    )

    parts = decode_finite(a)
    significand = parts.significand
    exponent = parts.exponent

    # Ensure exponent is even so that sqrt(sig * 2^exp) = sqrt(sig) * 2^(exp/2).
    if exponent % 2 != 0:
        significand <<= 1
        exponent -= 1
    half_exp = exponent >> 1

    steps.append(
        Step(
            "decode",
            "Decoded exact value; adjusted to even exponent.",
            {
                "significand": significand,
                "exponent": exponent,
                "half_exponent": half_exp,
                "value": f"{significand} × 2^{exponent}",
            },
        )
    )

    # Integer sqrt with p guard bits.  p = 2*significand_bits + 4 guarantees
    # that the perturbation used for the sticky bit is far below any ULP boundary.
    p = 2 * fmt.significand_bits + 4
    scaled = significand << (2 * p)
    q = isqrt(scaled)
    rem = scaled - q * q
    sticky = rem > 0

    steps.append(
        Step(
            "integer_sqrt",
            "Computed integer square root with guard bits.",
            {
                "guard_bits_p": p,
                "q": q,
                "remainder": rem,
                "exact": not sticky,
            },
        )
    )

    # Pack into the target format.
    # Exact value: q × 2^(half_exp − p).
    # Sticky: true value is q × 2^(half_exp − p) + tiny_irrational.
    # When sticky, replace numerator q with (q*2+1) so the rational passed to
    # pack_rational lies strictly between q/2^p and (q+1)/2^p; this ensures
    # _scaled_div_round always sees a non-zero remainder and never mistakes an
    # irrational result for an exact tie.
    if not sticky:
        result, flags = pack_scaled_integer(0, q, half_exp - p, fmt, rounding)
    else:
        numerator = q * 2 + 1
        denom_shift = p + 1 - half_exp
        if denom_shift >= 0:
            result, flags = pack_rational(0, numerator, 1 << denom_shift, fmt, rounding)
        else:
            result, flags = pack_rational(
                0, numerator << (-denom_shift), 1, fmt, rounding
            )
        flags = flags | {FloatingPointFlag.INEXACT}

    steps.append(
        Step("round", f"Rounded with {rounding.value}.", {"flags": _flag_values(flags)})
    )
    steps.append(_result_step(result))
    return _finish(result, steps, flags, op_name)


def _input_step(a: IEEEFloat, b: IEEEFloat) -> Step:
    return Step(
        "input",
        f"Inputs: a = {a.to_decimal()} ({a.to_bit_string()}), b = {b.to_decimal()} ({b.to_bit_string()})",
        {
            "a_decimal": a.to_decimal(),
            "b_decimal": b.to_decimal(),
            "a_bits": a.to_bit_string(),
            "b_bits": b.to_bit_string(),
        },
    )


def _result_step(result: IEEEFloat) -> Step:
    return Step(
        "result",
        f"Result: {result}",
        {"decimal": result.to_decimal(), "bit_string": result.to_bit_string()},
    )


def _finish(
    result: IEEEFloat,
    steps: List[Step],
    flags: Set[FloatingPointFlag],
    operation: str,
) -> ArithmeticResult:
    if not any(step.name == "result" for step in steps):
        steps.append(_result_step(result))
    return ArithmeticResult(
        result=result,
        steps=steps,
        precision_lost=FloatingPointFlag.INEXACT in flags,
        operation=operation,
        flags=set(flags),
    )


def _zero_sum(sign_a: int, sign_b: int, fmt, rounding: RoundingMode) -> IEEEFloat:
    if sign_a == sign_b:
        return _make_zero(sign_a, fmt)
    return (
        _make_zero(1, fmt)
        if rounding == RoundingMode.ROUND_TOWARD_NEG_INF
        else _make_zero(0, fmt)
    )


def _flag_values(flags: Set[FloatingPointFlag]) -> List[str]:
    return sorted(flag.value for flag in flags)
