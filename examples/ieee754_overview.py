"""
ieee754_overview — a guided tour of naive-ieee754

Run this script from the project root to see annotated output for every
major feature of the library.  It covers construction, bit inspection,
special values, arithmetic, rounding modes, custom formats, and
number-line visualization.

Usage:
    python examples/ieee754_overview.py

Requires matplotlib (already a project dependency).
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # interactive window; switch to "Agg" for headless

from naive_ieee754 import (
    Float8,
    Float16,
    Float32,
    Float64,
    custom_float,
    number_line,
)
from naive_ieee754.formats import (
    FLOAT8_FORMAT,
    FLOAT16_FORMAT,
    FLOAT32_FORMAT,
    FLOAT64_FORMAT,
)
from naive_ieee754.rounding import RoundingMode


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# -----------------------------------------------------------------
# 1. CONSTRUCTORS — four ways to arrive at the same bit pattern
# -----------------------------------------------------------------
section("1. Constructors")

# The most natural entry point: supply a decimal value and let the
# library handle the conversion (and rounding) to binary.
a = Float32.from_decimal(0.1)
print(f"from_decimal(0.1)      -> {a!r}")

# You can also describe the bits directly as a string.
# Spaces are ignored, so you can group sign / exponent / significand
# visually: "S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM"
b = Float32.from_bit_string("0 01111011 10011001100110011001101")
print(f"from_bit_string(...)   -> {b!r}")

# A raw integer pattern — what a debugger or memory dump would show.
# 0x3DCCCCCD is the standard IEEE 754 representation of 0.1 in Float32.
c = Float32.from_raw_int(0x3DCCCCCD)
print(f"from_raw_int(0x3DCC..) -> {c!r}")

# Field-by-field construction: sign bit, then exponent bits, then
# significand bits — all as plain Python lists of 0s and 1s.
d = Float32(
    0,
    [0, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
)
print(f"direct construction    -> {d!r}")

# All four should produce the same repr — they are four routes to
# the identical 32-bit pattern.


# -----------------------------------------------------------------
# 2. BIT INSPECTION — looking inside a float
# -----------------------------------------------------------------
section("2. Bit inspection")

# 1.5 is a good subject: it is exactly representable in binary
# (1.5 = 1 + 2^-1 = 1.1 in base 2), so there is no rounding to
# cloud the picture.
x = Float32.from_decimal(1.5)
print(f"value           : {x}")
print(f"repr            : {x!r}")
print(f"to_bit_string() : {x.to_bit_string()}")  # S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM
print(f"to_hex()        : {x.to_hex()}")
print()

# to_fields_dict() breaks the bit string into labelled fields and
# also decodes them (e.g. the stored exponent minus the bias).
fields = x.to_fields_dict()
for k, v in fields.items():
    print(f"  {k:<22} = {v}")


# -----------------------------------------------------------------
# 3. SPECIAL VALUES — NaN, +-Inf, +-0, subnormal
# -----------------------------------------------------------------
section("3. Special values")

# IEEE 754 reserves bit patterns where the exponent is all-zeros
# or all-ones for non-normal numbers.
specials = {
    "+Inf":     Float32.positive_infinity(),   # exp=11111111, sig=0
    "-Inf":     Float32.negative_infinity(),   # exp=11111111, sig=0, sign=1
    "NaN":      Float32.nan(),                 # exp=11111111, sig!=0
    "+0":       Float32.positive_zero(),       # exp=00000000, sig=0
    "-0":       Float32.negative_zero(),       # exp=00000000, sig=0, sign=1
    "subnormal": Float32.from_raw_int(0x00000001),  # exp=0, sig=1 (smallest)
}

for label, v in specials.items():
    print(
        f"  {label:<10} {str(v):<25}  kind={v.kind.value:<15}  bits={v.to_bit_string()}"
    )

# Classification predicates let you test which category a value
# falls into without inspecting its bits manually.
n = Float32.from_decimal(3.14)
print(
    f"\n3.14 -> is_normal={n.is_normal()}  is_subnormal={n.is_subnormal()}  "
    f"is_nan={n.is_nan()}  is_inf={n.is_inf()}"
)


# -----------------------------------------------------------------
# 4. ARITHMETIC WITH PYTHON OPERATORS
# -----------------------------------------------------------------
section("4. Arithmetic with Python operators")

# Standard Python operators are overloaded and use round-to-nearest-
# even by default (the IEEE 754 default rounding mode).
p = Float32.from_decimal(1.0)
q = Float32.from_decimal(2.0)

print(f"p = {p}   q = {q}")
print(f"p + q = {p + q}")
print(f"p - q = {p - q}")
print(f"p * q = {p * q}")
print(f"p / q = {p / q}")
print(f"-p    = {-p}")
print(f"|p-q| = {abs(p - q)}")


# -----------------------------------------------------------------
# 5. STEP-BY-STEP ARITHMETIC — the educational core
# -----------------------------------------------------------------
section("5. Step-by-step arithmetic — .add() / .mul() / .div() / .sqrt()")

# The verbose methods (.add, .mul, .div, .sqrt) return an
# ArithmeticResult object that records every intermediate step:
# alignment shifts, guard/round/sticky bits, the rounding decision,
# and which IEEE 754 exception flags were raised.
a = Float32.from_decimal(0.1)
b = Float32.from_decimal(0.2)

res = a.add(b)
print(res.explain())
print("\n--- Precision report ---")
print(res.precision_report())  # exact result vs. what Float32 actually stored

print("\n--- Square root of 2.0 ---")
two = Float32.from_decimal(2.0)
sqrt_res = two.sqrt()
print(sqrt_res.explain())


# -----------------------------------------------------------------
# 6. THE CLASSIC: why 0.1 + 0.2 != 0.3
# -----------------------------------------------------------------
section("6. Why 0.1 + 0.2 != 0.3?")

# 0.1 and 0.2 are non-terminating in binary (like 1/3 in decimal).
# Each is rounded when stored; the sum of two rounded values rarely
# equals the rounded value of their exact sum.
# Comparing across formats shows how precision affects the outcome.
for Fmt in (Float64, Float32, Float16, Float8):
    f01 = Fmt.from_decimal(0.1)
    f02 = Fmt.from_decimal(0.2)
    f03 = Fmt.from_decimal(0.3)
    total = f01 + f02

    print(f"\n{Fmt.__name__}:")
    print(f"  0.1 stored as  {f01.to_decimal()!r}  ({f01.to_bit_string()})")
    print(f"  0.2 stored as  {f02.to_decimal()!r}  ({f02.to_bit_string()})")
    print(f"  0.3 stored as  {f03.to_decimal()!r}  ({f03.to_bit_string()})")
    print(f"  0.1+0.2      = {total.to_decimal()!r}  ({total.to_bit_string()})")
    print(f"  0.1+0.2 == 0.3? -> {total == f03}")


# -----------------------------------------------------------------
# 7. ROUNDING MODES
# -----------------------------------------------------------------
section("7. Rounding modes")

# IEEE 754 defines five rounding modes.  The default (round-to-nearest-
# even, also called "banker's rounding") minimises accumulated bias
# over long sequences of operations.  The others are useful in
# interval arithmetic and other specialised contexts.
a = Float32.from_decimal(1.0)
b = Float32.from_decimal(3.0)

for mode in RoundingMode:
    r = a.div(b, rounding=mode)
    print(f"  1/3 with {mode.value:<35} -> {r.result.to_decimal()!r}")


# -----------------------------------------------------------------
# 8. CUSTOM FORMATS — custom_float()
# -----------------------------------------------------------------
section("8. Custom formats")

# custom_float() builds a new float class with any exponent/significand
# width.  Small formats are ideal for teaching: you can enumerate all
# their representable values and reason about them by hand.
Mini = custom_float(exponent_bits=4, significand_bits=3)
print(f"Mini = custom_float(4e + 3sig)  ->  {Mini.FORMAT}")

m = Mini.from_decimal(1.5)
print(f"Mini(1.5) = {m!r}  ->  decimal={m.to_decimal()}")

# Comparing how the same value is stored across formats makes the
# precision trade-off concrete.
for Fmt in (Float8, Float16, Float32, Float64):
    v = Fmt.from_decimal(0.1)
    print(f"  {Fmt.__name__:<8}  0.1 stored as  {v.to_decimal()!r}")


# -----------------------------------------------------------------
# 9. VISUALIZATION — number_line()
# -----------------------------------------------------------------
section("9. Visualization: number_line()")

# number_line() plots every representable value (or a dense sample
# for large formats) as a tick on a number line.  The key insight:
# floats are NOT uniformly distributed — they are denser near zero
# and sparser toward the extremes because each binade [2^n, 2^(n+1))
# holds the same count of values but twice the width.

# Float8 — three views stacked vertically
fig1, (ax1a, ax1b, ax1c) = plt.subplots(3, 1, figsize=(12, 7))
number_line(FLOAT8_FORMAT, ax=ax1a, log_scale=True)
ax1a.set_title("Float8 — full range (log scale)")
number_line(FLOAT8_FORMAT, ax=ax1b, log_scale=False)
ax1b.set_title("Float8 — full range (linear scale)")
# Zoom into [1, 8): three consecutive binades side by side.
# Notice that [1, 2) is twice as dense as [2, 4), twice as dense
# as [4, 8) — same number of representable values, wider interval.
number_line(FLOAT8_FORMAT, ax=ax1c, log_scale=False, x_range=(1, 8))
ax1c.set_xlim(right=8)
ax1c.set_title("Float8 — [1, 8) (linear scale)")
fig1.tight_layout()

# Float16 — same structure, higher resolution
fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(12, 7))
number_line(FLOAT16_FORMAT, ax=ax2a, log_scale=True)
ax2a.set_title("Float16 — full range (log scale)")
number_line(FLOAT16_FORMAT, ax=ax2b, log_scale=False)
ax2b.set_title("Float16 — full range (linear scale)")
number_line(FLOAT16_FORMAT, ax=ax2c, log_scale=False, x_range=(1, 8))
ax2c.set_xlim(right=8)
ax2c.set_title("Float16 — [1, 8) (linear scale)")
fig2.tight_layout()

# Float16 — subnormal region: values between 0 and min_positive_normal.
# Unlike normal numbers (log distribution), subnormals are linearly
# spaced — the exponent is fixed at zero so the grid no longer scales.
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 7))
number_line(
    FLOAT16_FORMAT,
    ax=axes3[0],
    log_scale=False,
    x_range=(-FLOAT16_FORMAT.min_positive_normal, FLOAT16_FORMAT.min_positive_normal),
)
axes3[0].set_title(
    "Float16 — subnormal domain [-min_normal, +min_normal) (linear scale)"
)
number_line(
    FLOAT16_FORMAT,
    ax=axes3[1],
    log_scale=True,
    x_range=(0, FLOAT16_FORMAT.min_positive_normal),
)
axes3[1].set_title("Float16 — positive subnormal region (0, min_normal) (log scale)")
number_line(
    FLOAT16_FORMAT,
    ax=axes3[2],
    log_scale=False,
    x_range=(0, FLOAT16_FORMAT.min_positive_normal),
)
axes3[2].set_title("Float16 — positive subnormal region [0, min_normal) (linear scale)")
fig3.tight_layout()

# Float8 vs Float16: consecutive unit-width binades side by side.
# Each column is a binade of width 1; each row is a format.
# More significand bits -> more ticks per binade.
_delta = 1
_fig4_formats = [FLOAT8_FORMAT, FLOAT16_FORMAT]
_fig4_ranges = [(1, 1 + _delta), (2, 2 + _delta), (4, 4 + _delta)]
fig4, axes4 = plt.subplots(2, 3, figsize=(12, 6))
for row, fmt in enumerate(_fig4_formats):
    for col, x_range in enumerate(_fig4_ranges):
        number_line(
            fmt,
            ax=axes4[row, col],
            log_scale=False,
            x_range=x_range,
        )
        axes4[row, col].set_title(f"{fmt.name}  [{x_range[0]}, {x_range[1]})")
fig4.tight_layout()

# Float32 vs Float64: narrow windows (delta = 1e-6) to make the
# precision gap visible.  Float32 has a handful of values; Float64
# has tens of thousands — in the same microscopic interval.
_delta = 1e-6
_fig5_formats = [FLOAT32_FORMAT, FLOAT64_FORMAT]
_fig5_ranges = [(1, 1 + _delta), (2, 2 + _delta), (4, 4 + _delta)]
fig5, axes5 = plt.subplots(2, 3, figsize=(12, 6))
for row, fmt in enumerate(_fig5_formats):
    for col, x_range in enumerate(_fig5_ranges):
        number_line(
            fmt,
            samples_per_exponent=2**30,
            ax=axes5[row, col],
            log_scale=False,
            x_range=x_range,
        )
        axes5[row, col].set_title(f"{fmt.name}  [{x_range[0]}, {x_range[1]})")
fig5.tight_layout()

plt.show()
print("\nTour complete.")
