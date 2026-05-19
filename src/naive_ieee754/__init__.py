"""naive-ieee754 — an educational IEEE 754 floating-point library.

Ever wondered why ``0.1 + 0.2`` gives ``0.30000000000000004``?  This library
won't fix that — but it will let you watch it happen in slow motion.

The "naive" in the name is intentional.  This implementation stores every bit
in a plain Python list and does all arithmetic with explicit loops.  It is not
fast.  It is not memory-efficient.  It is not something you would ever use in
production code.  It is a magnifying glass for understanding what IEEE 754 is
actually doing behind the scenes.

Quick start::

    from naive_ieee754 import Float32, Float8, number_line

    # Convert a decimal to IEEE 754 and inspect the bits
    a = Float32.from_decimal(0.1)
    print(repr(a))
    # Float32(sign=0 exp=[01111011] sig=[10011001100110011001101])

    # Arithmetic with step-by-step explanation
    b = Float32.from_decimal(0.2)
    result = a.add(b)
    print(result.explain())
    print(result.precision_report())

    # Visualize all representable Float8 values on a number line
    number_line(Float8.FORMAT)

See the README for more examples.
"""

from __future__ import annotations

from .arithmetic import ArithmeticResult, Step
from .core import IEEEFloat
from .flags import FloatingPointFlag
from .floats import Float8, Float16, Float32, Float64, custom_float
from .formats import (
    FLOAT16_FORMAT,
    FLOAT32_FORMAT,
    FLOAT64_FORMAT,
    FLOAT8_FORMAT,
    FloatFormat,
)
from .rounding import RoundingMode
from .special import SpecialKind
from .visualization import PlotInfo, number_line


__all__ = [
    # Core class
    "IEEEFloat",
    # Concrete formats
    "Float8",
    "Float16",
    "Float32",
    "Float64",
    # Format definitions
    "FloatFormat",
    "FLOAT8_FORMAT",
    "FLOAT16_FORMAT",
    "FLOAT32_FORMAT",
    "FLOAT64_FORMAT",
    "custom_float",
    # Arithmetic result types
    "ArithmeticResult",
    "Step",
    # Enums
    "FloatingPointFlag",
    "RoundingMode",
    "SpecialKind",
    # Visualization
    "number_line",
    "PlotInfo",
]
