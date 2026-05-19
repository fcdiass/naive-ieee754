"""Tests for the number_line visualization (no GUI)."""

import math
import pytest

from naive_ieee754 import number_line
from naive_ieee754.formats import FLOAT8_FORMAT, FLOAT16_FORMAT, FLOAT32_FORMAT

# Force a non-interactive backend before any matplotlib import so that tests
# run correctly in headless environments (CI, servers without a display).
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _MPL_AVAILABLE, reason="matplotlib not installed")


def test_number_line_float8():
    """number_line for Float8 should run without error (no window)."""
    fig, ax = plt.subplots()
    number_line(FLOAT8_FORMAT, ax=ax)
    plt.close(fig)


def test_number_line_float16():
    """number_line for Float16 should enumerate all values without crashing."""
    fig, ax = plt.subplots()
    number_line(FLOAT16_FORMAT, ax=ax)
    plt.close(fig)


def test_number_line_float32_sampled():
    """number_line for Float32 should use sampling (no crash)."""
    fig, ax = plt.subplots()
    number_line(FLOAT32_FORMAT, samples_per_exponent=4, ax=ax)
    plt.close(fig)


def test_number_line_positive_only():
    """x_range with xmin >= 0 should show only positive values."""
    fig, ax = plt.subplots()
    number_line(FLOAT8_FORMAT, x_range=(0.0, math.inf), ax=ax)
    plt.close(fig)


def test_number_line_custom_title():
    """Custom title should appear in the axes title."""
    fig, ax = plt.subplots()
    number_line(FLOAT8_FORMAT, title="my custom title", ax=ax)
    assert ax.get_title() == "my custom title"
    plt.close(fig)


def test_number_line_returns_plot_info():
    """number_line must return a PlotInfo with sensible fields."""
    from naive_ieee754 import PlotInfo

    fig, ax = plt.subplots()
    info = number_line(FLOAT8_FORMAT, x_range=(1.0, 8.0), ax=ax, show_info=False)
    plt.close(fig)

    assert isinstance(info, PlotInfo)
    assert info.fmt is FLOAT8_FORMAT
    assert info.repr_count is not None and info.repr_count > 0
    assert info.plotted > 0


def test_plot_info_source_exact_small_format():
    """Small formats (≤16 bits) must always report source='exact'."""
    fig, ax = plt.subplots()
    info = number_line(FLOAT8_FORMAT, ax=ax, show_info=False)
    plt.close(fig)
    assert info.source == "exact"
    assert info.step is None


def test_plot_info_source_exact_in_range():
    """Float32 with a tiny x_range (few representable values) uses exact enumeration."""
    # [1.0, 1.0 + 20 ULPs) has exactly 20 representable Float32 values,
    # well below the default samples_per_exponent=64.
    twenty_ulps = 20 * FLOAT32_FORMAT.eps
    fig, ax = plt.subplots()
    info = number_line(FLOAT32_FORMAT, x_range=(1.0, 1.0 + twenty_ulps), ax=ax, show_info=False)
    plt.close(fig)
    assert info.source == "exact in range"
    assert info.step is None
    assert info.plotted == 20


def test_plot_info_source_sampled():
    """Float32 over its full range must use sampling."""
    fig, ax = plt.subplots()
    info = number_line(FLOAT32_FORMAT, samples_per_exponent=4, ax=ax, show_info=False)
    plt.close(fig)
    assert info.source == "sampled"
    assert info.step is not None and info.step > 1


def test_plot_info_repr_count_none_for_unbounded_range():
    """Unbounded range (default ±∞) must return repr_count=None."""
    fig, ax = plt.subplots()
    info = number_line(FLOAT32_FORMAT, samples_per_exponent=4, ax=ax, show_info=False)
    plt.close(fig)
    assert info.repr_count is None


def test_show_info_prints_by_default(capsys):
    """show_info=True (default) must print a non-empty line to stdout."""
    fig, ax = plt.subplots()
    number_line(FLOAT8_FORMAT, ax=ax)
    plt.close(fig)
    assert capsys.readouterr().out.strip() != ""


def test_show_info_false_suppresses_output(capsys):
    """show_info=False must produce no stdout output."""
    fig, ax = plt.subplots()
    number_line(FLOAT8_FORMAT, ax=ax, show_info=False)
    plt.close(fig)
    assert capsys.readouterr().out == ""
