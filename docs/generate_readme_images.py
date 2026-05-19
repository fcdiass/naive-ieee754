"""Generate the images used in README.md.

Run from the repository root:
    python docs/generate_readme_images.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from naive_ieee754.formats import FLOAT16_FORMAT, FLOAT32_FORMAT, FLOAT64_FORMAT
from naive_ieee754.visualization import number_line

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _total_finite(fmt) -> int:
    """Exact count of finite values (including ±0) for an IEEE 754 format."""
    max_pos = ((fmt.max_stored_exponent - 1) << fmt.significand_bits) | (
        (1 << fmt.significand_bits) - 1
    )
    return (max_pos + 1) * 2


def _fmt_count(n: int) -> str:
    if n < 1_000_000:
        return f"{n:,}"
    return f"{n:.3e}"


IMAGES = [
    ("float16_density.png", FLOAT16_FORMAT, 64),
    ("float32_density.png", FLOAT32_FORMAT, 64),
    ("float64_density.png", FLOAT64_FORMAT, 32),
]

for filename, fmt, spe in IMAGES:
    print(f"\n{fmt.name}...")
    fig, ax = plt.subplots(figsize=(14, 2.6))
    fig.patch.set_facecolor("white")

    info = number_line(fmt, ax=ax, samples_per_exponent=spe, show_info=False)

    total = _total_finite(fmt)
    if info.source == "sampled":
        source_note = f"sampled — {info.plotted:,} plotted, 1 in every {info.step:,} significand values"
    else:
        source_note = f"exact — all {info.plotted:,} finite values plotted"

    ax.set_title(
        f"{fmt.name}  |  {_fmt_count(total)} finite values  ({source_note})  |  "
        f"{fmt.total_bits} bits ({fmt.exponent_bits}e + {fmt.significand_bits}sig)",
        fontsize=9,
    )

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")

print("\nDone.")
