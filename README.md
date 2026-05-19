# naive-ieee754

> Ever wondered why `0.1 + 0.2` gives `0.30000000000000004`?
> This library won't fix that — but it will let you watch it happen in slow motion.

`naive-ieee754` is an educational Python library for exploring IEEE 754 floating-point representation.  It stores every number explicitly as three separate fields — **sign**, **exponent bits**, and **significand bits** — as plain Python lists, and implements all arithmetic using explicit bit-by-bit loops.

The "naive" in the name is intentional.  This is not fast.  It is not memory-efficient.  It would be an awful choice for production code.  It is a magnifying glass for understanding what your CPU is _actually_ doing when you type `1.5 + 2.3`.

---

## Explore

Two resources cover the full feature set of the library, side by side:

| Format | File | Best for |
|--------|------|----------|
| Script | [`examples/ieee754_overview.py`](https://github.com/fcdiass/naive-ieee754/blob/main/examples/ieee754_overview.py) | Running in a terminal, reading annotated source |
| Notebook | [`notebooks/ieee754_overview.ipynb`](https://github.com/fcdiass/naive-ieee754/blob/main/notebooks/ieee754_overview.ipynb) | Interactive exploration in VS Code or JupyterLab |

Both walk through the same topics in the same order: constructors, bit inspection,
special values, step-by-step arithmetic, the 0.1 + 0.2 gotcha, rounding modes,
custom formats, and number-line visualization.

To run the script:

```bash
python examples/ieee754_overview.py
```

To open the notebook (requires dev dependencies):

```bash
uv sync --group dev
jupyter lab notebooks/ieee754_overview.ipynb
```

---

## Why does this exist?

Floating-point arithmetic is one of those topics where everyone nods along in class and then spends the next decade slowly discovering that `0.1` is not `0.1`, that `-0 == 0` but they are _different values_, and that `NaN != NaN` is perfectly valid behavior and not a bug in your comparison logic.

`naive-ieee754` makes all of this visible.  You can inspect every bit, trace every rounding decision, and plot every representable number on a number line.  Once you stare at the density gradient of Float8 values for a while, the weirdness starts to make a strange kind of sense.

---

## Installation

```bash
pip install naive-ieee754
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add naive-ieee754
```

---

## Quick start

```python
from naive_ieee754 import Float32, Float8, number_line

# Convert a decimal to IEEE 754 and inspect the bit fields
a = Float32.from_decimal(0.1)
print(repr(a))
# Float32(sign=0 exp=[01111011] sig=[10011001100110011001101])

print(a.to_bit_string())
# 0 01111011 10011001100110011001101

print(a.to_hex())
# 3DCCCCCD
```

### Step-by-step arithmetic

```python
b = Float32.from_decimal(0.2)
result = a.add(b)          # verbose; returns an ArithmeticResult
print(result.explain())    # step-by-step breakdown of the addition
print(result.precision_report())
# exact (Python float64): 0.30000000447034836
# represented (Float32): 0.30000001192092896
# absolute error: 7.450580596923828e-09
# relative error: 2.48e-08
```

Using the `+` operator works too and returns a plain `IEEEFloat`:

```python
c = a + b
print(str(c))
# 0.30000001192092896 (Float32)
```

### Special values

```python
from naive_ieee754 import Float32

print(Float32.positive_infinity())   # +Inf (Float32)
print(Float32.negative_zero())       # -0 (Float32)
print(Float32.nan())                 # NaN (Float32)

# IEEE 754 quirks in action
nan = Float32.nan()
print(nan == nan)    # False  (NaN is never equal to anything, including itself)

pz = Float32.positive_zero()
nz = Float32.negative_zero()
print(pz == nz)      # True   (IEEE 754: +0 == -0)
print(repr(pz))      # Float32(sign=0 exp=[00000000] sig=[00000000000000000000000])
print(repr(nz))      # Float32(sign=1 exp=[00000000] sig=[00000000000000000000000])
```

### Number line visualization

```python
from naive_ieee754 import Float8, number_line
from naive_ieee754.formats import FLOAT32_FORMAT

# Show every finite Float8 value (all 238 of them)
number_line(Float8.FORMAT)

# Sample Float32 values — 4 billion is too many to plot
number_line(FLOAT32_FORMAT, samples_per_exponent=8)
```

The plot uses a symmetric log scale so you can see how IEEE 754 values are
distributed across the full ±∞ range. Float16 is plotted exactly; Float32 and
Float64 are sampled — the title and legend report how many values exist and
what fraction is shown.

**Float16** — 63,488 finite values, all plotted (exact):

![Float16 full range on symlog scale](https://raw.githubusercontent.com/fcdiass/naive-ieee754/main/docs/images/float16_density.png)

**Float32** — ~4.28 × 10⁹ finite values, sampled:

![Float32 full range on symlog scale](https://raw.githubusercontent.com/fcdiass/naive-ieee754/main/docs/images/float32_density.png)

**Float64** — ~1.84 × 10¹⁹ finite values, sampled:

![Float64 full range on symlog scale](https://raw.githubusercontent.com/fcdiass/naive-ieee754/main/docs/images/float64_density.png)

### Custom formats

Want a 9-bit float with 3 exponent bits and 5 significand bits?  Sure, why not:

```python
from naive_ieee754 import custom_float, number_line

MyFloat = custom_float(exponent_bits=3, significand_bits=5)
x = MyFloat.from_decimal(1.5)
print(repr(x))
# Custom(3e+5sig)(sign=0 exp=[011] sig=[10000])

number_line(MyFloat.FORMAT)
```

---

## Supported formats

| Class    | Bits | Exponent | Significand | Bias  | Max value        |
|----------|------|----------|-------------|-------|------------------|
| `Float8`  |  8   |  4       |  3          |   7   | 240              |
| `Float16` | 16   |  5       | 10          |  15   | 65504            |
| `Float32` | 32   |  8       | 23          | 127   | ~3.4 × 10³⁸      |
| `Float64` | 64   | 11       | 52          | 1023  | ~1.8 × 10³⁰⁸     |

Custom formats can be created with `custom_float(exponent_bits, significand_bits)`.

---

## Rounding modes

By default all operations use **round-to-nearest-even** (IEEE 754's default).
You can pass a different `RoundingMode` to the verbose `.add()`, `.mul()`, etc.:

```python
from naive_ieee754 import Float32, RoundingMode

a = Float32.from_decimal(1.0)
b = Float32.from_decimal(3.0)
result = a.div(b, rounding=RoundingMode.ROUND_TOWARD_ZERO)
print(result.explain())
```

Available modes: `ROUND_TO_NEAREST_EVEN`, `ROUND_TOWARD_ZERO`,
`ROUND_TOWARD_POS_INF`, `ROUND_TOWARD_NEG_INF`.

Verbose arithmetic results also expose IEEE 754 status flags:

```python
from naive_ieee754 import FloatingPointFlag

result = Float32.from_decimal(1.0).div(Float32.from_decimal(3.0))
print(FloatingPointFlag.INEXACT in result.flags)
# True
```

---

## Subnormal numbers

Subnormal numbers (values very close to zero with stored exponent = 0) are
supported in conversion and arithmetic.  The library uses gradual underflow:
tiny exact results are rounded into the subnormal range when possible, and
inexact tiny results raise the `UNDERFLOW` and `INEXACT` flags in verbose
arithmetic results.

---

## License

MIT — see [LICENSE](https://github.com/fcdiass/naive-ieee754/blob/main/LICENSE).

**Author:** Francisco Corrêa Dias — [@fcdiass](https://github.com/fcdiass)
