"""Plot convergence metrics from an effective_probs progress file.

This script reads the text file produced by `error_propagation_simulation`
(`effective_probs_<platform>_<timestamp>.txt`) and plots four curves versus
iteration:
- X probability
- Y probability
- Z probability
- Bias

Example:
    python -m pauli_distribution.plot_progress_file \
        --progress-file effective_probs_superconducting_20260329.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt


def _parse_progress_file(
    progress_file: Path,
) -> Tuple[List[int], List[Optional[int]], List[float], List[float], List[float], List[float]]:
    """Parse iteration, optional generated samples, X, Y, Z, and bias columns."""
    iterations: List[int] = []
    generated_samples: List[Optional[int]] = []
    x_probs: List[float] = []
    y_probs: List[float] = []
    z_probs: List[float] = []
    bias_values: List[float] = []

    with progress_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            row = next(csv.reader([line]))
            if len(row) < 6:
                continue

            try:
                iteration = int(row[0])
                x_value = float(row[2])
                y_value = float(row[3])
                z_value = float(row[4])
                bias_value = float(row[5])
                sample_value: Optional[int] = None
                if len(row) >= 13 and row[12].strip() != "":
                    sample_value = int(row[12])
            except ValueError:
                # Skip malformed or non-numeric data rows.
                continue

            iterations.append(iteration)
            generated_samples.append(sample_value)
            x_probs.append(x_value)
            y_probs.append(y_value)
            z_probs.append(z_value)
            bias_values.append(bias_value)

    if not iterations:
        raise ValueError(f"No data rows were parsed from: {progress_file}")

    return iterations, generated_samples, x_probs, y_probs, z_probs, bias_values


def _build_output_path_with_flags(
    progress_file: Path,
    iter_start: int | None,
    iter_end: int | None,
    smooth_window: int,
    bias_log_scale: bool,
    x_axis: str,
    manual_sigfig: int | None = None,
) -> Path:
    stem = progress_file.stem
    parts = ["plots"]
    if iter_start is not None or iter_end is not None:
        start_label = "min" if iter_start is None else str(iter_start)
        end_label = "max" if iter_end is None else str(iter_end)
        parts.append(f"iter_{start_label}-{end_label}")
    if smooth_window > 1:
        parts.append(f"smooth_{smooth_window}")
    if bias_log_scale:
        parts.append("biaslog")
    if x_axis == "samples":
        parts.append("x_samples")
    if manual_sigfig is not None:
        parts.append(f"sigfig{manual_sigfig}")
    suffix = "_".join(parts)
    return progress_file.with_name(f"{stem}_{suffix}.svg")


def _moving_average(values: List[float], window: int) -> List[float]:
    """Centered moving average that ignores non-finite values in each window."""
    if window <= 1:
        return list(values)

    n = len(values)
    half = window // 2
    smoothed: List[float] = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_values = [v for v in values[start:end] if math.isfinite(v)]
        if window_values:
            smoothed.append(sum(window_values) / len(window_values))
        else:
            smoothed.append(math.nan)
    return smoothed


def _sanitize_for_log(values: List[float]) -> List[float]:
    """Replace non-positive/non-finite values with NaN for log-scale plotting."""
    return [v if (math.isfinite(v) and v > 0.0) else math.nan for v in values]


def _parse_counts_totals(counts_file: Path) -> List[int]:
    """Parse cumulative total observation counts from counts JSONL file."""
    totals: List[int] = []
    with counts_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if "counts" in data and isinstance(data["counts"], dict):
                counts_obj = data["counts"]
            else:
                counts_obj = data
            if not all(str(pauli_type) in counts_obj or pauli_type in counts_obj for pauli_type in (0, 1, 2, 3)):
                # Skip metadata rows that do not carry Pauli counts.
                continue
            try:
                total = 0
                for pauli_type in (0, 1, 2, 3):
                    total += int(counts_obj.get(str(pauli_type), counts_obj.get(pauli_type, 0)))
            except (TypeError, ValueError):
                continue
            totals.append(total)
    if not totals:
        raise ValueError(f"No valid count rows parsed from: {counts_file}")
    return totals


def _parse_counts_cumulative(counts_file: Path) -> List[Tuple[int, int, int, int]]:
    """Parse cumulative (I, X, Y, Z) counts per JSONL row."""
    cumulative: List[Tuple[int, int, int, int]] = []
    with counts_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if "counts" in data and isinstance(data["counts"], dict):
                counts_obj = data["counts"]
            else:
                counts_obj = data
            if not all(str(pauli_type) in counts_obj or pauli_type in counts_obj for pauli_type in (0, 1, 2, 3)):
                # Skip metadata rows that do not carry Pauli counts.
                continue
            try:
                c_i = int(counts_obj.get("0", counts_obj.get(0, 0)))
                c_x = int(counts_obj.get("1", counts_obj.get(1, 0)))
                c_y = int(counts_obj.get("2", counts_obj.get(2, 0)))
                c_z = int(counts_obj.get("3", counts_obj.get(3, 0)))
            except (TypeError, ValueError):
                continue
            cumulative.append((c_i, c_x, c_y, c_z))
    if not cumulative:
        raise ValueError(f"No valid cumulative count rows parsed from: {counts_file}")
    return cumulative


def _align_counts_to_progress_length(rows: List[Any], progress_len: int, label: str) -> List[Any]:
    """Align counts-derived rows to progress rows by tail-trimming extras."""
    if len(rows) < progress_len:
        raise ValueError(
            f"{label} rows do not align with parsed progress rows. "
            f"{label}_rows={len(rows)}, progress_rows={progress_len}"
        )
    if len(rows) > progress_len:
        return rows[-progress_len:]
    return rows


def _linear_slope_ci(
    x_values: List[int],
    y_values: List[float],
    z_score: float,
) -> Tuple[float, float, float]:
    """Compute OLS slope and normal CI for slope."""
    n = len(x_values)
    if n < 3:
        return math.nan, math.nan, math.nan

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    s_xx = sum((x - x_mean) ** 2 for x in x_values)
    if s_xx <= 0.0:
        return math.nan, math.nan, math.nan

    s_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    slope = s_xy / s_xx
    intercept = y_mean - slope * x_mean
    residual_ss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(x_values, y_values))
    dof = n - 2
    if dof <= 0:
        return slope, math.nan, math.nan
    sigma2 = residual_ss / dof
    se_slope = math.sqrt(max(0.0, sigma2 / s_xx))
    return slope, slope - z_score * se_slope, slope + z_score * se_slope


def _window_summary_from_delta_counts(
    cumulative_counts: List[Tuple[int, int, int, int]],
    start_index: int,
    end_index: int,
    z_score: float,
) -> dict[str, float]:
    """Estimate window statistics from independent count increments."""
    start_base = cumulative_counts[start_index - 1] if start_index > 0 else (0, 0, 0, 0)
    end_counts = cumulative_counts[end_index]
    d_i = end_counts[0] - start_base[0]
    d_x = end_counts[1] - start_base[1]
    d_y = end_counts[2] - start_base[2]
    d_z = end_counts[3] - start_base[3]
    n_window = d_i + d_x + d_y + d_z
    if n_window <= 0:
        nan_stats = {
            "n_window": 0.0,
            "x": math.nan,
            "x_low": math.nan,
            "x_high": math.nan,
            "y": math.nan,
            "y_low": math.nan,
            "y_high": math.nan,
            "z": math.nan,
            "z_low": math.nan,
            "z_high": math.nan,
            "i": math.nan,
            "i_low": math.nan,
            "i_high": math.nan,
            "bias": math.nan,
            "bias_low": math.nan,
            "bias_high": math.nan,
            "x_se": math.nan,
            "y_se": math.nan,
            "z_se": math.nan,
            "bias_se": math.nan,
        }
        return nan_stats

    n = float(n_window)
    p_x = d_x / n
    p_y = d_y / n
    p_z = d_z / n
    p_i = d_i / n
    s_xy = p_x + p_y
    bias = p_z / s_xy if s_xy > 0.0 else math.inf

    se_x = math.sqrt(max(0.0, p_x * (1.0 - p_x) / n))
    se_y = math.sqrt(max(0.0, p_y * (1.0 - p_y) / n))
    se_z = math.sqrt(max(0.0, p_z * (1.0 - p_z) / n))
    se_i = math.sqrt(max(0.0, p_i * (1.0 - p_i) / n))

    if s_xy > 0.0 and math.isfinite(bias):
        var_bias = p_z * (p_z + s_xy) / (n * (s_xy ** 3))
        se_bias = math.sqrt(max(0.0, var_bias))
    else:
        se_bias = math.nan

    return {
        "n_window": float(n_window),
        "x": p_x,
        "x_low": max(0.0, p_x - z_score * se_x),
        "x_high": min(1.0, p_x + z_score * se_x),
        "y": p_y,
        "y_low": max(0.0, p_y - z_score * se_y),
        "y_high": min(1.0, p_y + z_score * se_y),
        "z": p_z,
        "z_low": max(0.0, p_z - z_score * se_z),
        "z_high": min(1.0, p_z + z_score * se_z),
        "i": p_i,
        "i_low": max(0.0, p_i - z_score * se_i),
        "i_high": min(1.0, p_i + z_score * se_i),
        "bias": bias,
        "bias_low": (max(0.0, bias - z_score * se_bias) if math.isfinite(se_bias) else math.nan),
        "bias_high": (bias + z_score * se_bias if math.isfinite(se_bias) else math.nan),
        "x_se": se_x,
        "y_se": se_y,
        "z_se": se_z,
        "bias_se": se_bias,
    }


def _difference_ci(
    first_value: float,
    first_se: float,
    second_value: float,
    second_se: float,
    z_score: float,
) -> Tuple[float, float, float]:
    """Compute (second - first) estimate with normal CI from independent SEs."""
    if not (math.isfinite(first_value) and math.isfinite(second_value) and math.isfinite(first_se) and math.isfinite(second_se)):
        return math.nan, math.nan, math.nan
    delta = second_value - first_value
    delta_se = math.sqrt(max(0.0, first_se * first_se + second_se * second_se))
    return delta, delta - z_score * delta_se, delta + z_score * delta_se


def _round_to_ci_precision(value: float, low: float, high: float) -> float:
    """Round a value to the precision implied by CI half-width."""
    if not (math.isfinite(value) and math.isfinite(low) and math.isfinite(high)):
        return value
    half_width = abs(high - low) / 2.0
    if half_width <= 0.0 or not math.isfinite(half_width):
        return value
    decimals = int(-math.floor(math.log10(half_width)))
    return round(value, decimals)


def _quantize_series_from_ci(values: List[float], lows: List[float], highs: List[float]) -> List[float]:
    """Quantize each point using its own CI-derived precision."""
    return [_round_to_ci_precision(v, lo, hi) for v, lo, hi in zip(values, lows, highs)]


def _round_to_n_sigfigs(value: float, n_sigfigs: int) -> float:
    """Round a value to n significant figures."""
    if not math.isfinite(value) or value == 0.0:
        return value
    if n_sigfigs < 1:
        return value
    decimals = n_sigfigs - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, decimals)


def _quantize_series_from_sigfigs(values: List[float], n_sigfigs: int) -> List[float]:
    """Quantize all values to n significant figures."""
    return [_round_to_n_sigfigs(v, n_sigfigs) for v in values]


def _compute_probability_cis(
    x_probs: List[float],
    y_probs: List[float],
    z_probs: List[float],
    bias_values: List[float],
    totals: List[int],
    z_score: float,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Compute normal-approximation CI bounds for X/Y/Z probabilities and bias."""
    x_low: List[float] = []
    x_high: List[float] = []
    y_low: List[float] = []
    y_high: List[float] = []
    z_low: List[float] = []
    z_high: List[float] = []
    b_low: List[float] = []
    b_high: List[float] = []

    for x_prob, y_prob, z_prob, bias_value, total in zip(x_probs, y_probs, z_probs, bias_values, totals):
        n = float(total)
        if n <= 0.0:
            x_low.append(math.nan)
            x_high.append(math.nan)
            y_low.append(math.nan)
            y_high.append(math.nan)
            z_low.append(math.nan)
            z_high.append(math.nan)
            b_low.append(math.nan)
            b_high.append(math.nan)
            continue

        se_x = math.sqrt(max(0.0, x_prob * (1.0 - x_prob) / n))
        se_y = math.sqrt(max(0.0, y_prob * (1.0 - y_prob) / n))
        se_z = math.sqrt(max(0.0, z_prob * (1.0 - z_prob) / n))

        x_low.append(max(0.0, x_prob - z_score * se_x))
        x_high.append(min(1.0, x_prob + z_score * se_x))
        y_low.append(max(0.0, y_prob - z_score * se_y))
        y_high.append(min(1.0, y_prob + z_score * se_y))
        z_low.append(max(0.0, z_prob - z_score * se_z))
        z_high.append(min(1.0, z_prob + z_score * se_z))

        s_xy = x_prob + y_prob
        if s_xy <= 0.0 or not math.isfinite(bias_value):
            b_low.append(math.nan)
            b_high.append(math.nan)
            continue

        var_bias = z_prob * (z_prob + s_xy) / (n * (s_xy ** 3))
        se_bias = math.sqrt(max(0.0, var_bias))
        b_low.append(max(0.0, bias_value - z_score * se_bias))
        b_high.append(bias_value + z_score * se_bias)

    return x_low, x_high, y_low, y_high, z_low, z_high, b_low, b_high


def _filter_iteration_range(
    iterations: List[int],
    generated_samples: List[Optional[int]],
    x_probs: List[float],
    y_probs: List[float],
    z_probs: List[float],
    bias_values: List[float],
    iter_start: int | None,
    iter_end: int | None,
) -> Tuple[List[int], List[Optional[int]], List[float], List[float], List[float], List[float]]:
    if iter_start is None and iter_end is None:
        return iterations, generated_samples, x_probs, y_probs, z_probs, bias_values

    filtered_iterations: List[int] = []
    filtered_samples: List[Optional[int]] = []
    filtered_x: List[float] = []
    filtered_y: List[float] = []
    filtered_z: List[float] = []
    filtered_bias: List[float] = []

    for it, sample_count, x, y, z, b in zip(iterations, generated_samples, x_probs, y_probs, z_probs, bias_values):
        if iter_start is not None and it < iter_start:
            continue
        if iter_end is not None and it > iter_end:
            continue
        filtered_iterations.append(it)
        filtered_samples.append(sample_count)
        filtered_x.append(x)
        filtered_y.append(y)
        filtered_z.append(z)
        filtered_bias.append(b)

    if not filtered_iterations:
        raise ValueError(
            "No rows remain after applying iteration range filter: "
            f"start={iter_start}, end={iter_end}."
        )

    return filtered_iterations, filtered_samples, filtered_x, filtered_y, filtered_z, filtered_bias


def plot_progress_file(
    progress_file: Path,
    output_path: Path | None = None,
    show: bool = False,
    smooth_window: int = 1,
    bias_log_scale: bool = False,
    iter_start: int | None = None,
    iter_end: int | None = None,
    x_axis: str = "iteration",
    counts_file: Path | None = None,
    error_bars: bool = False,
    error_z: float = 1.96,
    window_summary: bool = False,
    advanced_summary: bool = False,
    summary_report: Path | None = None,
    quality_check: bool = False,
    qc_bias_conv_threshold: float = 2e-4,
    qc_bias_ci_halfwidth: float = 1e-3,
    qc_require_split_overlap_zero: bool = True,
    sigfig_from_ci: bool = False,
    sigfig_overlay: bool = True,
    manual_sigfig: int | None = None,
) -> Path:
    """Create and save a 2x2 plot grid for X, Y, Z probabilities and bias."""
    all_iterations, all_generated_samples, all_x_probs, all_y_probs, all_z_probs, all_bias_values = _parse_progress_file(progress_file)

    selected_indices = [
        idx
        for idx, iteration in enumerate(all_iterations)
        if (iter_start is None or iteration >= iter_start) and (iter_end is None or iteration <= iter_end)
    ]
    if not selected_indices:
        raise ValueError(
            "No rows remain after applying iteration range filter: "
            f"start={iter_start}, end={iter_end}."
        )

    iterations = [all_iterations[idx] for idx in selected_indices]
    generated_samples = [all_generated_samples[idx] for idx in selected_indices]
    x_probs = [all_x_probs[idx] for idx in selected_indices]
    y_probs = [all_y_probs[idx] for idx in selected_indices]
    z_probs = [all_z_probs[idx] for idx in selected_indices]
    bias_values = [all_bias_values[idx] for idx in selected_indices]

    iterations, generated_samples, x_probs, y_probs, z_probs, bias_values = _filter_iteration_range(
        iterations,
        generated_samples,
        x_probs,
        y_probs,
        z_probs,
        bias_values,
        iter_start,
        iter_end,
    )

    if output_path is None:
        output_path = _build_output_path_with_flags(
            progress_file,
            iter_start=iter_start,
            iter_end=iter_end,
            smooth_window=smooth_window,
            bias_log_scale=bias_log_scale,
            x_axis=x_axis,
            manual_sigfig=manual_sigfig,
        )
    output_path = output_path.expanduser().resolve(strict=False)

    normalized_x_axis = x_axis.strip().lower()
    if normalized_x_axis not in {"iteration", "samples"}:
        raise ValueError("x_axis must be one of: 'iteration', 'samples'.")

    if normalized_x_axis == "samples":
        if any(sample_count is None for sample_count in generated_samples):
            raise ValueError(
                "Selected x-axis 'samples', but this progress file does not have "
                "Generated_Samples for all rows. Use x_axis='iteration' or regenerate data."
            )
        x_values = [int(sample_count) for sample_count in generated_samples]
        x_label = "Generated samples"
    else:
        x_values = iterations
        x_label = "Iteration"

    x_low = x_high = y_low = y_high = z_low = z_high = b_low = b_high = None
    x_sig = y_sig = z_sig = b_sig = None
    window_stats: Optional[dict[str, float]] = None
    advanced_stats: Optional[dict[str, float]] = None
    summary_text: Optional[str] = None
    if error_bars or sigfig_from_ci:
        if counts_file is None:
            requirement = "--error-bars" if error_bars else "--sigfig-from-ci"
            raise ValueError(f"{requirement} requires --counts-file.")
        if not counts_file.exists() or counts_file.is_dir():
            raise FileNotFoundError(f"Counts file not found or is not a file: {counts_file}")
        totals_all = _align_counts_to_progress_length(
            _parse_counts_totals(counts_file),
            len(all_iterations),
            "counts",
        )
        totals_filtered = [totals_all[idx] for idx in selected_indices]
        x_low, x_high, y_low, y_high, z_low, z_high, b_low, b_high = _compute_probability_cis(
            x_probs,
            y_probs,
            z_probs,
            bias_values,
            totals_filtered,
            error_z,
        )
        if sigfig_from_ci:
            x_sig = _quantize_series_from_ci(x_probs, x_low, x_high)
            y_sig = _quantize_series_from_ci(y_probs, y_low, y_high)
            z_sig = _quantize_series_from_ci(z_probs, z_low, z_high)
            b_sig = _quantize_series_from_ci(bias_values, b_low, b_high)

    if manual_sigfig is not None:
        if manual_sigfig < 1:
            raise ValueError("--manual-sigfig must be >= 1.")
        x_sig = _quantize_series_from_sigfigs(x_probs, manual_sigfig)
        y_sig = _quantize_series_from_sigfigs(y_probs, manual_sigfig)
        z_sig = _quantize_series_from_sigfigs(z_probs, manual_sigfig)
        b_sig = _quantize_series_from_sigfigs(bias_values, manual_sigfig)

    if advanced_summary and not window_summary:
        window_summary = True

    if window_summary:
        if counts_file is None:
            raise ValueError("--window-summary requires --counts-file.")
        if not counts_file.exists() or counts_file.is_dir():
            raise FileNotFoundError(f"Counts file not found or is not a file: {counts_file}")
        cumulative_counts = _align_counts_to_progress_length(
            _parse_counts_cumulative(counts_file),
            len(all_iterations),
            "counts",
        )
        start_index = selected_indices[0]
        end_index = selected_indices[-1]
        window_stats = _window_summary_from_delta_counts(
            cumulative_counts,
            start_index,
            end_index,
            error_z,
        )

        slope_x, slope_x_low, slope_x_high = _linear_slope_ci(x_values, x_probs, error_z)
        slope_y, slope_y_low, slope_y_high = _linear_slope_ci(x_values, y_probs, error_z)
        slope_z, slope_z_low, slope_z_high = _linear_slope_ci(x_values, z_probs, error_z)
        slope_b, slope_b_low, slope_b_high = _linear_slope_ci(x_values, bias_values, error_z)
        if window_stats["n_window"] <= 0:
            summary_text = (
                "Window summary n=0\n"
                "No incremental samples in selected window.\n"
                f"dX/dx={slope_x:.3e} [{slope_x_low:.3e}, {slope_x_high:.3e}]\n"
                f"dY/dx={slope_y:.3e} [{slope_y_low:.3e}, {slope_y_high:.3e}]\n"
                f"dZ/dx={slope_z:.3e} [{slope_z_low:.3e}, {slope_z_high:.3e}]\n"
                f"dBias/dx={slope_b:.3e} [{slope_b_low:.3e}, {slope_b_high:.3e}]"
            )
        else:
            summary_text = (
                f"Window summary n={int(window_stats['n_window'])}\n"
                f"X={window_stats['x']:.6g} [{window_stats['x_low']:.6g}, {window_stats['x_high']:.6g}]\n"
                f"Y={window_stats['y']:.6g} [{window_stats['y_low']:.6g}, {window_stats['y_high']:.6g}]\n"
                f"Z={window_stats['z']:.6g} [{window_stats['z_low']:.6g}, {window_stats['z_high']:.6g}]\n"
                f"Bias={window_stats['bias']:.6g} [{window_stats['bias_low']:.6g}, {window_stats['bias_high']:.6g}]\n"
                f"dX/dx={slope_x:.3e} [{slope_x_low:.3e}, {slope_x_high:.3e}]\n"
                f"dY/dx={slope_y:.3e} [{slope_y_low:.3e}, {slope_y_high:.3e}]\n"
                f"dZ/dx={slope_z:.3e} [{slope_z_low:.3e}, {slope_z_high:.3e}]\n"
                f"dBias/dx={slope_b:.3e} [{slope_b_low:.3e}, {slope_b_high:.3e}]"
            )

        if advanced_summary:
            half = len(selected_indices) // 2
            if half >= 1 and len(selected_indices) - half >= 1:
                first_start = selected_indices[0]
                first_end = selected_indices[half - 1]
                second_start = selected_indices[half]
                second_end = selected_indices[-1]
                first_stats = _window_summary_from_delta_counts(cumulative_counts, first_start, first_end, error_z)
                second_stats = _window_summary_from_delta_counts(cumulative_counts, second_start, second_end, error_z)

                dx, dx_low, dx_high = _difference_ci(first_stats["x"], first_stats["x_se"], second_stats["x"], second_stats["x_se"], error_z)
                dy, dy_low, dy_high = _difference_ci(first_stats["y"], first_stats["y_se"], second_stats["y"], second_stats["y_se"], error_z)
                dz, dz_low, dz_high = _difference_ci(first_stats["z"], first_stats["z_se"], second_stats["z"], second_stats["z_se"], error_z)
                db, db_low, db_high = _difference_ci(first_stats["bias"], first_stats["bias_se"], second_stats["bias"], second_stats["bias_se"], error_z)

                advanced_stats = {
                    "first_n": first_stats["n_window"],
                    "second_n": second_stats["n_window"],
                    "delta_x": dx,
                    "delta_x_low": dx_low,
                    "delta_x_high": dx_high,
                    "delta_y": dy,
                    "delta_y_low": dy_low,
                    "delta_y_high": dy_high,
                    "delta_z": dz,
                    "delta_z_low": dz_low,
                    "delta_z_high": dz_high,
                    "delta_bias": db,
                    "delta_bias_low": db_low,
                    "delta_bias_high": db_high,
                    "slope_x": slope_x,
                    "slope_x_low": slope_x_low,
                    "slope_x_high": slope_x_high,
                    "slope_y": slope_y,
                    "slope_y_low": slope_y_low,
                    "slope_y_high": slope_y_high,
                    "slope_z": slope_z,
                    "slope_z_low": slope_z_low,
                    "slope_z_high": slope_z_high,
                    "slope_bias": slope_b,
                    "slope_bias_low": slope_b_low,
                    "slope_bias_high": slope_b_high,
                }

                summary_text = (
                    f"{summary_text}\n"
                    f"Split delta (second-first)\n"
                    f"n1={int(first_stats['n_window'])}, n2={int(second_stats['n_window'])}\n"
                    f"dX={dx:.3e} [{dx_low:.3e}, {dx_high:.3e}]\n"
                    f"dY={dy:.3e} [{dy_low:.3e}, {dy_high:.3e}]\n"
                    f"dZ={dz:.3e} [{dz_low:.3e}, {dz_high:.3e}]\n"
                    f"dBias={db:.3e} [{db_low:.3e}, {db_high:.3e}]"
                )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_x, ax_y, ax_z, ax_b = axes.ravel()

    if smooth_window > 1:
        x_smooth = _moving_average(x_probs, smooth_window)
        y_smooth = _moving_average(y_probs, smooth_window)
        z_smooth = _moving_average(z_probs, smooth_window)
        bias_base = _sanitize_for_log(bias_values) if bias_log_scale else bias_values
        bias_smooth = _moving_average(bias_base, smooth_window)

        ax_x.plot(x_values, x_probs, color="tab:blue", linewidth=1.0, alpha=0.35, label="Raw")
        ax_x.plot(x_values, x_smooth, color="tab:blue", linewidth=2.0, label=f"Smoothed (w={smooth_window})")

        ax_y.plot(x_values, y_probs, color="tab:orange", linewidth=1.0, alpha=0.35, label="Raw")
        ax_y.plot(x_values, y_smooth, color="tab:orange", linewidth=2.0, label=f"Smoothed (w={smooth_window})")

        ax_z.plot(x_values, z_probs, color="tab:green", linewidth=1.0, alpha=0.35, label="Raw")
        ax_z.plot(x_values, z_smooth, color="tab:green", linewidth=2.0, label=f"Smoothed (w={smooth_window})")

        ax_b.plot(x_values, bias_base, color="tab:red", linewidth=1.0, alpha=0.35, label="Raw")
        ax_b.plot(x_values, bias_smooth, color="tab:red", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
    else:
        bias_base = _sanitize_for_log(bias_values) if bias_log_scale else bias_values
        raw_alpha = 0.4 if (sigfig_from_ci or manual_sigfig is not None) else 1.0
        ax_x.plot(x_values, x_probs, color="tab:blue", linewidth=1.8, alpha=raw_alpha)
        ax_y.plot(x_values, y_probs, color="tab:orange", linewidth=1.8, alpha=raw_alpha)
        ax_z.plot(x_values, z_probs, color="tab:green", linewidth=1.8, alpha=raw_alpha)
        ax_b.plot(x_values, bias_base, color="tab:red", linewidth=1.8, alpha=raw_alpha)

    if sigfig_from_ci and sigfig_overlay and x_sig is not None and y_sig is not None and z_sig is not None and b_sig is not None:
        ax_x.plot(x_values, x_sig, color="tab:blue", linewidth=1.6, linestyle="--", alpha=0.95, label="CI-rounded")
        ax_y.plot(x_values, y_sig, color="tab:orange", linewidth=1.6, linestyle="--", alpha=0.95, label="CI-rounded")
        ax_z.plot(x_values, z_sig, color="tab:green", linewidth=1.6, linestyle="--", alpha=0.95, label="CI-rounded")
        ax_b.plot(x_values, b_sig, color="tab:red", linewidth=1.6, linestyle="--", alpha=0.95, label="CI-rounded")
    elif manual_sigfig is not None and x_sig is not None and y_sig is not None and z_sig is not None and b_sig is not None:
        label_text = f"{manual_sigfig} sig-fig"
        ax_x.plot(x_values, x_sig, color="tab:blue", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)
        ax_y.plot(x_values, y_sig, color="tab:orange", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)
        ax_z.plot(x_values, z_sig, color="tab:green", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)
        ax_b.plot(x_values, b_sig, color="tab:red", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)

    ax_x.set_title("X probability")
    ax_x.set_ylabel("X probability")
    ax_x.grid(True, alpha=0.3)
    if error_bars and x_low is not None and x_high is not None:
        ax_x.fill_between(x_values, x_low, x_high, color="tab:blue", alpha=0.18, linewidth=0)
    if window_summary and window_stats is not None and window_stats["n_window"] > 0:
        ax_x.axhline(window_stats["x"], color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.9)
        ax_x.axhspan(window_stats["x_low"], window_stats["x_high"], color="tab:blue", alpha=0.08)

    ax_y.set_title("Y probability")
    ax_y.set_ylabel("Y probability")
    ax_y.grid(True, alpha=0.3)
    if error_bars and y_low is not None and y_high is not None:
        ax_y.fill_between(x_values, y_low, y_high, color="tab:orange", alpha=0.18, linewidth=0)
    if window_summary and window_stats is not None and window_stats["n_window"] > 0:
        ax_y.axhline(window_stats["y"], color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.9)
        ax_y.axhspan(window_stats["y_low"], window_stats["y_high"], color="tab:orange", alpha=0.08)

    ax_z.set_title("Z probability")
    ax_z.set_ylabel("Z probability")
    ax_z.set_xlabel(x_label)
    ax_z.grid(True, alpha=0.3)
    if error_bars and z_low is not None and z_high is not None:
        ax_z.fill_between(x_values, z_low, z_high, color="tab:green", alpha=0.18, linewidth=0)
    if window_summary and window_stats is not None and window_stats["n_window"] > 0:
        ax_z.axhline(window_stats["z"], color="tab:green", linestyle="--", linewidth=1.2, alpha=0.9)
        ax_z.axhspan(window_stats["z_low"], window_stats["z_high"], color="tab:green", alpha=0.08)

    ax_b.set_title("Bias")
    ax_b.set_ylabel("Bias")
    ax_b.set_xlabel(x_label)
    ax_b.grid(True, alpha=0.3)
    if bias_log_scale:
        ax_b.set_yscale("log")
    if error_bars and b_low is not None and b_high is not None:
        ax_b.fill_between(x_values, b_low, b_high, color="tab:red", alpha=0.18, linewidth=0)
    if window_summary and window_stats is not None and window_stats["n_window"] > 0 and math.isfinite(window_stats["bias"]):
        ax_b.axhline(window_stats["bias"], color="tab:red", linestyle="--", linewidth=1.2, alpha=0.9)
        if math.isfinite(window_stats["bias_low"]) and math.isfinite(window_stats["bias_high"]):
            ax_b.axhspan(window_stats["bias_low"], window_stats["bias_high"], color="tab:red", alpha=0.08)

    if smooth_window > 1 or (sigfig_from_ci and sigfig_overlay):
        for axis in (ax_x, ax_y, ax_z, ax_b):
            axis.legend(loc="best", fontsize=8)

    iter_label_start = "min" if iter_start is None else str(iter_start)
    iter_label_end = "max" if iter_end is None else str(iter_end)
    flags_label = (
        f"iter_range={iter_label_start}-{iter_label_end}, "
        f"smooth_window={smooth_window}, bias_log_scale={bias_log_scale}, x_axis={normalized_x_axis}, "
        f"error_bars={error_bars}, sigfig_from_ci={sigfig_from_ci}, window_summary={window_summary}, "
        f"advanced_summary={advanced_summary}, error_z={error_z}"
    )
    fig.suptitle(f"Convergence metrics from {progress_file.name}")
    if window_summary and summary_text is not None:
        fig.text(
            0.995,
            0.985,
            summary_text,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.5"},
        )
    fig.text(0.5, 0.01, f"Flags: {flags_label}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, format='svg')
    except FileNotFoundError:
        # Fallback for edge cases with relative path resolution in some shells.
        fallback_output = (Path.cwd() / output_path.name).resolve()
        fallback_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fallback_output, format='svg')
        output_path = fallback_output

    if show:
        plt.show()
    else:
        plt.close(fig)

    if summary_report is not None:
        quality_block: Optional[dict[str, Any]] = None
        if quality_check:
            bias_convergence_value = math.nan
            if len(bias_values) >= 2:
                bias_convergence_value = abs(bias_values[-1] - bias_values[-2])

            bias_ci_halfwidth = math.nan
            if window_stats is not None and math.isfinite(window_stats.get("bias_high", math.nan)) and math.isfinite(window_stats.get("bias_low", math.nan)):
                bias_ci_halfwidth = (window_stats["bias_high"] - window_stats["bias_low"]) / 2.0

            split_overlap_zero = False
            if advanced_stats is not None:
                low = advanced_stats.get("delta_bias_low", math.nan)
                high = advanced_stats.get("delta_bias_high", math.nan)
                if math.isfinite(low) and math.isfinite(high):
                    split_overlap_zero = (low <= 0.0 <= high)

            checks = {
                "bias_convergence": {
                    "value": bias_convergence_value,
                    "threshold": qc_bias_conv_threshold,
                    "pass": (math.isfinite(bias_convergence_value) and bias_convergence_value <= qc_bias_conv_threshold),
                },
                "bias_ci_halfwidth": {
                    "value": bias_ci_halfwidth,
                    "threshold": qc_bias_ci_halfwidth,
                    "pass": (math.isfinite(bias_ci_halfwidth) and bias_ci_halfwidth <= qc_bias_ci_halfwidth),
                },
                "split_delta_bias_overlaps_zero": {
                    "value": split_overlap_zero,
                    "required": qc_require_split_overlap_zero,
                    "pass": (split_overlap_zero if qc_require_split_overlap_zero else True),
                },
            }

            overall_pass = all(check["pass"] for check in checks.values())
            quality_block = {
                "enabled": True,
                "overall_pass": overall_pass,
                "checks": checks,
            }

        report_payload: dict[str, Any] = {
            "progress_file": str(progress_file),
            "iter_start": iter_start,
            "iter_end": iter_end,
            "x_axis": normalized_x_axis,
            "error_z": error_z,
            "window_summary": window_stats,
            "advanced_summary": advanced_stats,
            "quality_check": quality_block,
            "sigfig_from_ci": sigfig_from_ci,
        }
        summary_report = summary_report.expanduser().resolve(strict=False)
        summary_report.parent.mkdir(parents=True, exist_ok=True)
        with summary_report.open("w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot X/Y/Z probability and bias vs iteration from progress_file."
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        required=True,
        help="Path to effective_probs_<platform>_<timestamp>.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <progress_file_stem>_plots.svg)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the file.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average window size (<=1 disables smoothing).",
    )
    parser.add_argument(
        "--bias-log-scale",
        action="store_true",
        help="Use logarithmic y-axis for the bias subplot.",
    )
    parser.add_argument(
        "--iter-start",
        type=int,
        default=None,
        help="Minimum iteration to include (inclusive).",
    )
    parser.add_argument(
        "--iter-end",
        type=int,
        default=None,
        help="Maximum iteration to include (inclusive).",
    )
    parser.add_argument(
        "--x-axis",
        choices=["iteration", "samples"],
        default="iteration",
        help="X-axis to use. 'samples' requires Generated_Samples column in the progress file.",
    )
    parser.add_argument(
        "--preset",
        choices=["basic", "uncertainty", "quality", "sigfig"],
        default="basic",
        help="Convenience preset that enables a recommended set of options.",
    )
    parser.add_argument(
        "--counts-file",
        type=Path,
        default=None,
        help="Path to running_counts_<platform>_<timestamp>.jsonl used for uncertainty bands.",
    )
    parser.add_argument(
        "--error-bars",
        action="store_true",
        help="Plot approximate 95% confidence bands (requires --counts-file).",
    )
    parser.add_argument(
        "--error-z",
        type=float,
        default=1.96,
        help="Z-score multiplier for uncertainty bands (default: 1.96).",
    )
    parser.add_argument(
        "--window-summary",
        action="store_true",
        help="Add statistically robust window summary from count increments (requires --counts-file).",
    )
    parser.add_argument(
        "--advanced-summary",
        action="store_true",
        help="Include split-window change (second-first) and trend summary in the annotation.",
    )
    parser.add_argument(
        "--summary-report",
        type=Path,
        default=None,
        help="Optional JSON path to write computed summary statistics.",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Evaluate pass/fail gates for estimate quality and include them in summary-report JSON.",
    )
    parser.add_argument(
        "--qc-bias-conv-threshold",
        type=float,
        default=2e-4,
        help="Quality gate: max allowed last-step |delta_bias| (default: 2e-4).",
    )
    parser.add_argument(
        "--qc-bias-ci-halfwidth",
        type=float,
        default=1e-3,
        help="Quality gate: max allowed bias CI half-width from window summary (default: 1e-3).",
    )
    parser.add_argument(
        "--qc-no-split-overlap-zero",
        action="store_true",
        help="Disable requiring split-window delta bias CI to overlap zero.",
    )
    parser.add_argument(
        "--sigfig-from-ci",
        action="store_true",
        help="Overlay CI-rounded trend values to emphasize significant digits (requires --counts-file).",
    )
    parser.add_argument(
        "--no-sigfig-overlay",
        action="store_true",
        help="Disable drawing the CI-rounded overlay line when --sigfig-from-ci is enabled.",
    )
    parser.add_argument(
        "--manual-sigfig",
        type=int,
        default=None,
        help="Overlay a line rounded to N significant figures (mutually exclusive with --sigfig-from-ci).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    progress_file = args.progress_file
    if not progress_file.exists():
        raise FileNotFoundError(f"Progress file not found: {progress_file}")
    if progress_file.is_dir():
        raise ValueError(
            "--progress-file points to a directory, not a file. "
            "Pass the full path to effective_probs_<platform>_<timestamp>.txt"
        )
    if args.iter_start is not None and args.iter_end is not None and args.iter_start > args.iter_end:
        raise ValueError("--iter-start must be <= --iter-end.")
    
    if args.sigfig_from_ci and args.manual_sigfig is not None:
        raise ValueError("--sigfig-from-ci and --manual-sigfig are mutually exclusive.")

    if args.preset == "uncertainty":
        args.error_bars = True
    elif args.preset == "quality":
        args.error_bars = True
        args.window_summary = True
        args.advanced_summary = True
        args.quality_check = True
    elif args.preset == "sigfig":
        args.error_bars = True
        args.sigfig_from_ci = True

    output_path = plot_progress_file(
        progress_file=progress_file,
        output_path=args.output,
        show=args.show,
        smooth_window=args.smooth_window,
        bias_log_scale=args.bias_log_scale,
        iter_start=args.iter_start,
        iter_end=args.iter_end,
        x_axis=args.x_axis,
        counts_file=args.counts_file,
        error_bars=args.error_bars,
        error_z=args.error_z,
        window_summary=args.window_summary,
        advanced_summary=args.advanced_summary,
        summary_report=args.summary_report,
        quality_check=args.quality_check,
        qc_bias_conv_threshold=args.qc_bias_conv_threshold,
        qc_bias_ci_halfwidth=args.qc_bias_ci_halfwidth,
        qc_require_split_overlap_zero=not args.qc_no_split_overlap_zero,
        sigfig_from_ci=args.sigfig_from_ci,
        sigfig_overlay=not args.no_sigfig_overlay,
        manual_sigfig=args.manual_sigfig,
    )
    print(f"Saved plots to: {output_path}")


if __name__ == "__main__":
    main()
