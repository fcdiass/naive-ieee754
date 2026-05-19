"""Number line visualization for IEEE 754 formats."""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .formats import FloatFormat

# matplotlib's coordinate transforms use float64 internally.  When the total
# x-axis span approaches float64's max (~1.8e308), transform matrices become
# singular due to overflow.  Half of float64 max is a safe plotting threshold.
_MATPLOTLIB_LINEAR_MAX = sys.float_info.max / 2  # ~8.98e307

# Formats with total_bits <= this threshold are fully enumerated.
# Float8 (8 bits): 256 patterns.  Float16 (16 bits): 65 536 patterns.
# Float32 (32 bits): 4 billion patterns → sample instead.
_ENUMERATE_THRESHOLD_BITS = 16

_PROGRESS_THRESHOLD = 100_000


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotInfo:
    """Summary of what number_line() computed and plotted.

    Attributes:
        fmt: The FloatFormat that was visualized.
        x_range: The half-open interval [xmin, xmax) that was plotted.
        source: How values were collected — ``"exact"``, ``"exact in range"``,
            or ``"sampled"``.
        repr_count: Exact count of representable finite values in x_range,
            or ``None`` when the range is unbounded (±∞).
        plotted: Number of values actually drawn on the axes (may be less than
            repr_count when sampling).
        step: Significand step used during sampling (1 in every *step* values
            was drawn).  ``None`` when source is exact.
    """

    fmt: FloatFormat
    x_range: Tuple[float, float]
    source: str
    repr_count: Optional[int]
    plotted: int
    step: Optional[int]

    def __str__(self) -> str:
        xmin, xmax = self.x_range
        range_str = f"[{_fmt_bound(xmin)}, {_fmt_bound(xmax)})"
        count_str = _fmt_count(self.repr_count) if self.repr_count is not None else "∞"
        if self.source == "sampled":
            mode_str = f"sampled — 1 in every {self.step:,} significand values"
        else:
            mode_str = self.source
        return (
            f"{self.fmt.name}  {range_str}  —  "
            f"{count_str} representable  —  "
            f"{self.plotted:,} plotted  ({mode_str})"
        )


# ---------------------------------------------------------------------------
# Direct bit-pattern decoding (skips IEEEFloat object construction)
# ---------------------------------------------------------------------------


def _decode_pos_pattern(
    pattern: int, p: int, bias: int, sig_mask: int, leading_bit: int
) -> float:
    """Decode an unsigned (stored_exp << p | sig) bit pattern to its float value.

    `pattern` encodes a positive finite value: the high bits are the stored
    exponent and the low `p` bits are the significand.  Subnormals
    (stored_exp == 0) and normals are handled in one branch each.  No
    IEEEFloat is constructed — this is what makes the hot loop fast.
    """
    sig_int = pattern & sig_mask
    stored_exp = pattern >> p
    if stored_exp == 0:
        # Subnormal: value = sig * 2^(1 - bias - p)
        return math.ldexp(float(sig_int), 1 - bias - p)
    # Normal: value = (1 << p | sig) * 2^(stored_exp - bias - p)
    return math.ldexp(float(leading_bit | sig_int), stored_exp - bias - p)


# ---------------------------------------------------------------------------
# Value collection
# ---------------------------------------------------------------------------


def _enumerate_all_values(fmt: FloatFormat) -> List[float]:
    """Return all finite representable values by exhaustive bit-pattern walk."""
    p = fmt.significand_bits
    bias = fmt.bias
    sig_mask = (1 << p) - 1
    leading_bit = 1 << p
    # Positive finite patterns: stored_exp in [0, max_stored_exponent - 1].
    max_pos_bits = ((fmt.max_stored_exponent - 1) << p) | sig_mask

    values: List[float] = []
    for pat in range(max_pos_bits + 1):
        v = _decode_pos_pattern(pat, p, bias, sig_mask, leading_bit)
        values.append(v)
        if pat != 0:  # avoid duplicating zero
            values.append(-v)
        else:
            values.append(-0.0)
    return sorted(values)


def _sample_values(
    fmt: FloatFormat, samples_per_exponent: int, xmin: float, xmax: float
) -> List[float]:
    """Return a representative sample of finite values for large formats.

    Uses direct math.ldexp decoding (no IEEEFloat instances) which is roughly
    two orders of magnitude faster than constructing a bit-list object per
    sample.

    The significand iteration is clipped to x_range at the binade level: only
    the significand integers whose decoded value falls in [xmin, xmax) are
    visited, so narrow ranges (e.g. Float64 over [1, 1.01)) are fast even
    when samples_per_exponent is large.
    """
    from tqdm import tqdm

    p = fmt.significand_bits
    bias = fmt.bias
    leading_bit = 1 << p
    max_significand = 1 << p
    step = max(1, max_significand // samples_per_exponent)

    signs = (0, 1) if xmin < 0 else (0,)

    def _ldexp_safe(exp: int) -> float:
        """math.ldexp(1.0, exp) but returns inf instead of raising on overflow."""
        try:
            return math.ldexp(1.0, exp)
        except OverflowError:
            return math.inf

    def _exp_in_range(sign: int, stored_exp: int) -> bool:
        # Binade for stored_exp:
        #   stored_exp == 0 → subnormal region [0, min_positive_normal)
        #   stored_exp >= 1 → [2^(stored_exp - bias), 2^(stored_exp - bias + 1))
        if stored_exp == 0:
            binade_lo = 0.0
            binade_hi = _ldexp_safe(1 - bias)
        else:
            binade_lo = _ldexp_safe(stored_exp - bias)
            binade_hi = _ldexp_safe(stored_exp - bias + 1)
        if sign == 0:
            return not (xmax <= binade_lo or xmin >= binade_hi)
        return not (xmin > -binade_lo or xmax <= -binade_hi)

    pairs = [
        (sign, exp)
        for sign in signs
        for exp in range(0, fmt.max_stored_exponent)  # include 0 (subnormals)
        if _exp_in_range(sign, exp)
    ]

    def _sig_bounds(sign: int, stored_exp: int) -> Tuple[int, int]:
        """Significand range [sig_lo, sig_hi) clipped to x_range for this binade.

        Uses the identity  value = (base | sig) * 2^actual_exp  to invert
        x_range bounds into significand integer bounds, so only the portion of
        the binade that overlaps [xmin, xmax) is visited.
        """
        if stored_exp == 0:
            actual_exp = 1 - bias - p
            binade_lo, binade_hi = 0.0, _ldexp_safe(1 - bias)
            base = 0
        else:
            actual_exp = stored_exp - bias - p
            binade_lo = _ldexp_safe(stored_exp - bias)
            binade_hi = _ldexp_safe(stored_exp - bias + 1)
            base = leading_bit

        # Effective absolute-value bounds for this binade/sign combination.
        if sign == 0:
            eff_lo = max(xmin, binade_lo)
            eff_hi = min(xmax, binade_hi)
        else:
            # Negative values: abs(v) in (-xmax, -xmin] clipped to binade.
            eff_lo = max(binade_lo, max(0.0, -xmax))
            eff_hi = min(binade_hi, -xmin)

        if eff_lo >= eff_hi:
            return 0, 0

        # sig = abs_value * 2^(-actual_exp) - base
        neg_ae = -actual_exp  # always a positive integer
        try:
            sig_lo = max(0, math.ceil(math.ldexp(eff_lo, neg_ae) - base))
        except OverflowError:
            sig_lo = 0
        try:
            sig_hi = min(max_significand, math.ceil(math.ldexp(eff_hi, neg_ae) - base))
        except OverflowError:
            sig_hi = max_significand

        return sig_lo, max(sig_lo, sig_hi)

    pair_bounds = [_sig_bounds(sign, exp) for sign, exp in pairs]
    total_steps = sum(len(range(lo, hi, step)) for lo, hi in pair_bounds)

    if total_steps > _PROGRESS_THRESHOLD:
        pbar = tqdm(
            total=total_steps,
            desc=f"Sampling {fmt.name}",
            unit=" values",
        )
        update_interval = max(1, total_steps // 100)
    else:
        pbar = None

    values: List[float] = []
    for (sign, stored_exp), (sig_lo, sig_hi) in zip(pairs, pair_bounds):
        if stored_exp == 0:
            actual_exp = 1 - bias - p
            base = 0  # no implicit leading bit
        else:
            actual_exp = stored_exp - bias - p
            base = leading_bit
        sign_factor = -1.0 if sign == 1 else 1.0

        pending = 0
        for sig_int in range(sig_lo, sig_hi, step):
            v = math.ldexp(float(base | sig_int), actual_exp)
            values.append(sign_factor * v if v != 0.0 else (-0.0 if sign else 0.0))
            if pbar is not None:
                pending += 1
                if pending == update_interval:
                    pbar.update(pending)
                    pending = 0
        if pbar is not None and pending:
            pbar.update(pending)

    if pbar is not None:
        pbar.close()

    return sorted(values)


def _enumerate_in_range(
    fmt: FloatFormat, xmin: float, xmax: float
) -> List[float]:
    """Return every representable finite value v with xmin <= v < xmax.

    Walks the IEEE 754 bit pattern in order (positive patterns are an
    order-preserving bijection with integers) and decodes each via
    `_decode_pos_pattern`.  Used only when the caller has already verified
    the count is bounded.
    """
    from ._binary import _FMT_TO_CLASS
    from .bits import bits_to_int

    p = fmt.significand_bits
    bias = fmt.bias
    sig_mask = (1 << p) - 1
    leading_bit = 1 << p
    max_pos_bits = ((fmt.max_stored_exponent - 1) << p) | sig_mask
    cls = _FMT_TO_CLASS[fmt]
    max_normal = fmt.max_normal

    def pos_bits_ge(x: float) -> int:
        """Smallest positive bit pattern whose value >= x (x >= 0)."""
        if x <= 0:
            return 0
        if x > max_normal:
            return max_pos_bits + 1
        f = cls.from_decimal(x)
        if f.is_inf():
            return max_pos_bits + 1
        bits = bits_to_int(f.exponent + f.significand)
        if f.to_decimal() < x:
            bits += 1
        return bits

    def pos_bits_lt(x: float) -> int:
        """Smallest positive bit pattern whose value >= x (for x > 0), i.e. an
        exclusive upper bound on patterns with value < x."""
        return pos_bits_ge(x)

    # --- Compute iteration bounds for both sides before allocating the bar ---
    pos_p_start = pos_p_end = 0
    if xmax > 0:
        lo = max(0.0, xmin)
        pos_p_start = pos_bits_ge(lo)
        pos_p_end = max_pos_bits + 1 if math.isinf(xmax) else pos_bits_lt(xmax)

    neg_p_start = neg_p_end = 0
    if xmin < 0:
        upper_excl = min(0.0, xmax)
        m_lo_excl = -upper_excl
        if math.isinf(xmin):
            m_hi_incl = math.inf
        else:
            m_hi_incl = -xmin

        if m_lo_excl == 0.0:
            neg_p_start = 1
        else:
            neg_p_start = pos_bits_ge(m_lo_excl)
            if neg_p_start <= max_pos_bits:
                v0 = _decode_pos_pattern(neg_p_start, p, bias, sig_mask, leading_bit)
                if v0 <= m_lo_excl:
                    neg_p_start += 1

        if math.isinf(m_hi_incl) or m_hi_incl >= max_normal:
            neg_p_end = max_pos_bits + 1
        else:
            p_candidate = pos_bits_ge(m_hi_incl)
            if p_candidate > max_pos_bits:
                neg_p_end = max_pos_bits + 1
            else:
                v_c = _decode_pos_pattern(p_candidate, p, bias, sig_mask, leading_bit)
                neg_p_end = p_candidate + 1 if v_c == m_hi_incl else p_candidate

    total = max(0, pos_p_end - pos_p_start) + max(0, neg_p_end - neg_p_start)

    from tqdm import tqdm

    if total > _PROGRESS_THRESHOLD:
        pbar = tqdm(total=total, desc=f"Enumerating {fmt.name}", unit=" values")
        update_interval = max(1, total // 200)
    else:
        pbar = None

    values: List[float] = []

    def _append_range(p_start: int, p_end: int, negate: bool) -> None:
        pending = 0
        for pat in range(p_start, p_end):
            v = _decode_pos_pattern(pat, p, bias, sig_mask, leading_bit)
            values.append(-v if negate else v)
            if pbar is not None:
                pending += 1
                if pending == update_interval:
                    pbar.update(pending)
                    pending = 0
        if pbar is not None and pending:
            pbar.update(pending)

    if xmax > 0:
        _append_range(pos_p_start, pos_p_end, negate=False)
    if xmin < 0:
        _append_range(neg_p_start, neg_p_end, negate=True)

    if pbar is not None:
        pbar.close()

    return sorted(values)


# ---------------------------------------------------------------------------
# Representable-value counting (exact via IEEE 754 bit-pattern ordering)
# ---------------------------------------------------------------------------


def _count_representable(xmin: float, xmax: float, fmt: FloatFormat) -> Optional[int]:
    """Count representable finite values (including ±0) in [xmin, xmax).

    Returns None when the range is unbounded (xmin or xmax is ±inf).

    For formats with ≤ 16 bits the count is exact via enumeration.
    For larger formats it uses the IEEE 754 bit-pattern ordering property:
    positive finite values (including +0 and subnormals) are in bijection
    with non-negative integers via their exponent+mantissa bit pattern, and
    the bijection is order-preserving.
    """
    if math.isinf(xmin) or math.isinf(xmax):
        return None

    from ._binary import _FMT_TO_CLASS
    from .special import SpecialKind
    from .bits import bits_to_int

    if fmt.total_bits <= _ENUMERATE_THRESHOLD_BITS:
        total = 0
        for p in range(2**fmt.total_bits):
            f = _FMT_TO_CLASS[fmt].from_raw_int(p)
            if f.kind in (
                SpecialKind.NAN,
                SpecialKind.POSITIVE_INF,
                SpecialKind.NEGATIVE_INF,
            ):
                continue
            if xmin <= f.to_decimal() < xmax:
                total += 1
        return total

    # --- Large formats: bit-pattern trick ---
    max_pos_bits = ((2**fmt.exponent_bits - 2) << fmt.significand_bits) | (
        2**fmt.significand_bits - 1
    )

    def _pos_bits(x: float) -> int:
        if x <= 0:
            return 0
        if x >= fmt.max_normal:
            return max_pos_bits
        f = _FMT_TO_CLASS[fmt].from_decimal(abs(x))
        return bits_to_int(f.exponent + f.significand)

    def _count_pos(lo: float, hi: float) -> int:
        if hi < 0 or lo > hi:
            return 0
        lo = max(0.0, lo)
        hi = min(hi, fmt.max_normal)

        p_lo = _pos_bits(lo)
        if lo > 0:
            f = _FMT_TO_CLASS[fmt].from_decimal(lo)
            if f.to_decimal() < lo:
                p_lo += 1

        p_hi = _pos_bits(hi)
        if hi > 0:
            f = _FMT_TO_CLASS[fmt].from_decimal(hi)
            if f.is_inf() or f.to_decimal() >= hi:
                p_hi -= 1

        return max(0, p_hi - p_lo + 1)

    count = 0
    if xmax >= 0:
        count += _count_pos(max(0.0, xmin), xmax)
    if xmin < 0:
        neg_abs_lo = abs(min(0.0, xmax))
        neg_abs_hi = abs(xmin)
        count += _count_pos(neg_abs_lo, neg_abs_hi)

    return count


# ---------------------------------------------------------------------------
# Title helpers
# ---------------------------------------------------------------------------


def _fmt_count(n: int) -> str:
    """Human-readable count: comma-separated below 1 M, scientific above."""
    if n < 1_000_000:
        return f"{n:,}"
    return f"{n:.3e}"


def _fmt_bound(x: float) -> str:
    """Format a range bound for the title."""
    if math.isinf(x):
        return "−∞" if x < 0 else "+∞"
    if x == int(x) and abs(x) < 1e6:
        return str(int(x))
    return f"{x:g}"


def _fmt_step(n: int) -> str:
    """Format a significand step count compactly (e.g. 4,194,304 → '4.19M')."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3g}M"
    if n >= 1_000:
        return f"{n / 1_000:.3g}K"
    return str(n)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def number_line(
    fmt: FloatFormat,
    samples_per_exponent: int = 64,
    highlight_subnormals: bool = True,
    log_scale: bool = True,
    title: Optional[str] = None,
    ax=None,
    x_range: Tuple[float, float] = (-math.inf, math.inf),
    show_info: bool = True,
) -> PlotInfo:
    """Plot representable finite values of a format on a number line.

    Args:
        fmt: The FloatFormat to visualize.
        samples_per_exponent: For large formats (>16 bits), how many mantissa
            values to sample per stored-exponent bucket.  Ignored for formats
            with ≤ 16 bits, which are always fully enumerated.  When the total
            representable count in the range is ≤ this value (and ≤
            _EXACT_ENUM_LIMIT), every value is enumerated exactly instead of
            sampled — set this high (e.g. 200_000) to force exact enumeration
            on small ranges.
        highlight_subnormals: If True, shade the subnormal region on log scale.
        log_scale: If True, use log (or symlog) scale on the x-axis.
        title: Custom plot title.  Auto-generated if None.
        ax: A matplotlib Axes to draw into.  If None, a new figure is created
            and plt.show() is called.  Pass an existing Axes to embed the plot.
        x_range: Half-open interval ``[xmin, xmax)`` of values to display,
            following Python's range convention (xmin included, xmax excluded).
            Defaults to ``(-inf, +inf)`` which shows the full range.  When
            xmin >= 0 only positive values are shown; when xmin < 0 negative
            values are shown as well.  The title reports the exact count of
            representable values in the interval (exact for Float8/Float16,
            via bit-pattern ordering for Float32/Float64).
        show_info: If True (default), print a one-line summary of how many
            values are representable in the range and whether exact enumeration
            or sampling was used.  Pass ``show_info=False`` to suppress output.

    Returns:
        PlotInfo with the exact representable count, plotted count, sampling
        mode, and step size.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If xmin > xmax.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for number line visualization.  "
            "Install it with: pip install matplotlib"
        ) from exc

    xmin, xmax = x_range
    if xmin > xmax:
        raise ValueError(f"x_range requires xmin <= xmax, got ({xmin}, {xmax})")

    show_negative = xmin < 0

    # --- Compute exact representable count for bounded ranges ---
    repr_count = _count_representable(xmin, xmax, fmt)  # None if unbounded

    # --- Collect values ---
    if fmt.total_bits <= _ENUMERATE_THRESHOLD_BITS:
        all_values = _enumerate_all_values(fmt)
        source = "exact"
        _step: Optional[int] = None
    elif repr_count is not None and repr_count <= samples_per_exponent:
        all_values = _enumerate_in_range(fmt, xmin, xmax)
        source = "exact in range"
        _step = None
    else:
        all_values = _sample_values(fmt, samples_per_exponent, xmin, xmax)
        source = "sampled"
        _step = max(1, (1 << fmt.significand_bits) // samples_per_exponent)

    # --- Filter to x_range (no-op for _enumerate_in_range, which is exact) ---
    is_default_range = math.isinf(xmin) and xmin < 0 and math.isinf(xmax) and xmax > 0
    if not is_default_range and source != "exact in range":
        all_values = [v for v in all_values if xmin <= v < xmax]

    positive_values = [v for v in all_values if v > 0]
    negative_values = [v for v in all_values if v < 0]
    show_zero = xmin <= 0.0 < xmax

    # --- Auto-switch to log scale when plotted range overflows matplotlib ---
    max_plotted = max((abs(v) for v in all_values if not math.isinf(v)), default=0.0)
    if not log_scale and max_plotted > _MATPLOTLIB_LINEAR_MAX:
        warnings.warn(
            f"{fmt.name}: plotted values reach {max_plotted:.2e}, which overflows "
            "matplotlib's float64 coordinate transforms on a linear axis. "
            "Switching to log_scale=True automatically.",
            stacklevel=2,
        )
        log_scale = True

    # --- Plot ---
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(16, 2.5))

    # Set the scale BEFORE adding data so matplotlib never builds linear
    # transforms for values that approach float64's representable limit.
    if log_scale:
        if show_negative and negative_values:
            # linthresh must be a normal float64 (not subnormal); subnormal
            # linthresh causes matplotlib's symlog to compute 1/linthresh = inf.
            _raw = (
                fmt.min_positive_subnormal * 2
                if fmt.min_positive_subnormal > 0
                else 1e-10
            )
            linthresh = max(_raw, 1e-300)
            ax.set_xscale("symlog", linthresh=linthresh)
        elif positive_values:
            ax.set_xscale("log")

        # For formats like Float64 whose max value is close to float64's own
        # limit, clip plotted values so autoscale margins never overflow.
        if max_plotted > _MATPLOTLIB_LINEAR_MAX:
            _safe = 1e307
            positive_values = [min(v, _safe) for v in positive_values]
            negative_values = [max(v, -_safe) for v in negative_values]

    _mode_label = (
        f"sampled (1 in {_fmt_step(_step)})"
        if source == "sampled"
        else "exact"
    )

    if positive_values:
        ax.scatter(
            positive_values,
            [0.0] * len(positive_values),
            marker="|",
            s=30,
            color="#2ecc71",
            alpha=0.7,
            linewidths=0.8,
            label=f"positive ({len(positive_values):,} · {_mode_label})",
            zorder=3,
        )

    if show_negative and negative_values:
        ax.scatter(
            negative_values,
            [0.0] * len(negative_values),
            marker="|",
            s=30,
            color="#e74c3c",
            alpha=0.7,
            linewidths=0.8,
            label=f"negative ({len(negative_values):,} · {_mode_label})",
            zorder=3,
        )

    if show_zero:
        ax.scatter([0], [0], marker="o", s=40, color="black", zorder=4, label="±0")

    # --- Subnormal region highlight ---
    if highlight_subnormals and fmt.min_positive_normal > 0:
        normal_boundary = fmt.min_positive_normal
        in_range = xmin < normal_boundary <= xmax
        in_range_neg = show_negative and xmin <= -normal_boundary < xmax

        # axvspan — skip when the subnormal boundary is so tiny relative to the
        # plotted range that matplotlib's log transform overflows (e.g. Float64).
        _draw_span = in_range
        if log_scale and in_range:
            _draw_span = max_plotted == 0.0 or (normal_boundary / max_plotted) >= 1e-300

        if _draw_span:
            ax.axvspan(
                0, normal_boundary, alpha=0.12, color="orange", zorder=1, label="subnormal region"
            )
            if in_range_neg:
                ax.axvspan(-normal_boundary, 0, alpha=0.12, color="orange", zorder=1)

        # axvline — always draw the boundary marker; only labelled when span is absent
        if in_range:
            ax.axvline(
                normal_boundary,
                color="orange",
                linewidth=1.0,
                linestyle="--",
                zorder=2,
                label=None if _draw_span else "subnormal boundary",
            )
        if in_range_neg:
            ax.axvline(-normal_boundary, color="orange", linewidth=1.0, linestyle="--", zorder=2)

    # --- Zero line ---
    if show_zero:
        ax.axvline(0, color="black", linewidth=0.8, zorder=2)

    # --- ±Inf annotations (only at truly open ends) ---
    if math.isinf(xmax) and xmax > 0:
        ax.annotate(
            "+∞",
            xy=(1.0, 0.5),
            xycoords="axes fraction",
            fontsize=9,
            ha="left",
            va="center",
            color="#2ecc71",
        )
    if show_negative and math.isinf(xmin) and xmin < 0:
        ax.annotate(
            "−∞",
            xy=(0.0, 0.5),
            xycoords="axes fraction",
            fontsize=9,
            ha="right",
            va="center",
            color="#e74c3c",
        )

    # --- Cosmetics ---
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plotted = len(positive_values) + len(negative_values) + (1 if show_zero else 0)

    # --- Title ---
    if title:
        ax.set_title(title, fontsize=10)
    elif is_default_range:
        auto_title = (
            f"{fmt.name} number line  |  {plotted:,} finite values plotted ({source})  |  "
            f"{fmt.total_bits} bits ({fmt.exponent_bits}e + {fmt.significand_bits}sig)"
        )
        ax.set_title(auto_title, fontsize=10)
    else:
        range_str = f"[{_fmt_bound(xmin)}, {_fmt_bound(xmax)})"
        count_str = _fmt_count(repr_count) if repr_count is not None else "?"
        plotted_note = (
            f"  |  {plotted:,} plotted ({source})"
            if fmt.total_bits > _ENUMERATE_THRESHOLD_BITS
            else ""
        )
        auto_title = (
            f"{fmt.name}  |  {range_str}  |  "
            f"{count_str} representable values{plotted_note}  |  "
            f"{fmt.total_bits} bits ({fmt.exponent_bits}e + {fmt.significand_bits}sig)"
        )
        ax.set_title(auto_title, fontsize=10)

    ax.legend(loc="best", fontsize=8, framealpha=0.7)

    if own_fig:
        plt.tight_layout()
        plt.show()

    # --- Build and optionally print result ---
    info = PlotInfo(
        fmt=fmt,
        x_range=(xmin, xmax),
        source=source,
        repr_count=repr_count,
        plotted=plotted,
        step=_step,
    )
    if show_info:
        print(info)
    return info



