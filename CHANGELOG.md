# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-19

### Added

- `Float8`, `Float16`, `Float32`, and `Float64` concrete float types backed by explicit bit lists (sign, exponent, significand)
- `custom_float()` factory for creating arbitrary-precision IEEE 754 formats beyond the four standard sizes
- `IEEEFloat` base class encapsulating the full IEEE 754 bit representation and conversion logic
- Step-by-step arithmetic operations (`.add()`, `.mul()`, `.div()`, `.sqrt()`) with `.explain()` to narrate each computational step
- `precision_report()` method reporting ULP error and relative error against Python's native `float`
- Support for all IEEE 754 special values: ±0, ±Inf, NaN, and subnormals with gradual underflow
- Five rounding modes (`ROUND_TO_NEAREST_EVEN`, `ROUND_TO_NEAREST_AWAY`, `ROUND_TOWARD_ZERO`, `ROUND_TOWARD_POSITIVE`, `ROUND_TOWARD_NEGATIVE`), defaulting to round-to-nearest-even per IEEE 754-2019
- `FloatingPointFlag`, `RoundingMode`, and `SpecialKind` enums for structured introspection
- `number_line()` visualization of all representable values for a given format, rendered via matplotlib
- Exhaustive test suite for `Float8` and `Float16`; sampled coverage for `Float32` and `Float64`
- Annotated terminal walkthrough in `examples/ieee754_overview.py`
- Interactive Jupyter notebook in `notebooks/ieee754_overview.ipynb`

[unreleased]: https://github.com/fcdiass/naive-ieee754/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fcdiass/naive-ieee754/releases/tag/v0.1.0
