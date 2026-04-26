#!/usr/bin/env python3
"""
Batch flatten JPEG STM topography exports.

The script treats the JPEG intensity as a height proxy. It cannot recover the
physical calibration or clipped values lost during JPEG/export, but it can remove
low-order scan/background trends in a conservative way.

Scientific safeguards used here:
  - Step edges and sharp defects are excluded from the background fit.
  - Smooth connected regions are treated as separate terrace-like components.
  - The fit includes one constant offset per component, so terrace height
    differences do not drive the background model.
  - Only the low-order polynomial background is subtracted. Component/terrace
    offsets are not subtracted.
  - Model selection prefers no correction, then plane, then quadratic only when
    the residual improvement is large enough.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr
from skimage.filters import sobel
from skimage.morphology import binary_dilation, disk


EPS = 1.0e-12


@dataclass
class MaskResult:
    feature_mask: np.ndarray
    edge_mask: np.ndarray
    defect_mask: np.ndarray
    labels: np.ndarray
    thresholds: dict[str, float]
    component_areas: list[int]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    name: str
    degree: int
    background: np.ndarray
    score_mad: float
    residual_p95_abs: float
    background_range_98: float
    n_poly_terms: int
    fit_pixels: int
    n_regions: int
    bic_like: float
    solver_iterations: int | None = None
    solver_condition_estimate: float | None = None
    success: bool = True
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservatively flatten one DAT/TIFF/JPEG STM topography file or all files in a folder."
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing .dat/.tif/.tiff/.jpg/.jpeg files, or a single supported file.",
    )
    parser.add_argument("--output_folder", required=True, help="Folder for flattened outputs.")

    parser.add_argument("--edge_percentile", type=float, default=90.0)
    parser.add_argument("--edge_mad_multiplier", type=float, default=5.0)
    parser.add_argument("--edge_dilate_px", type=int, default=7)
    parser.add_argument("--gradient_sigma", type=float, default=1.0)

    parser.add_argument("--defect_percentile", type=float, default=98.5)
    parser.add_argument("--defect_mad_multiplier", type=float, default=8.0)
    parser.add_argument("--defect_filter_size", type=int, default=5)
    parser.add_argument("--defect_dilate_px", type=int, default=1)

    parser.add_argument("--min_component_pixels", type=int, default=160)
    parser.add_argument("--min_component_fraction", type=float, default=0.0025)
    parser.add_argument("--max_components", type=int, default=80)
    parser.add_argument("--min_fit_fraction", type=float, default=0.08)
    parser.add_argument("--max_fit_pixels", type=int, default=120000)

    parser.add_argument("--plane_min_relative_improvement", type=float, default=0.05)
    parser.add_argument("--quadratic_min_relative_improvement", type=float, default=0.08)
    parser.add_argument("--quadratic_from_none_min_relative_improvement", type=float, default=0.12)
    parser.add_argument("--min_abs_improvement", type=float, default=0.002)
    parser.add_argument("--min_background_range", type=float, default=0.005)

    parser.add_argument("--random_seed", type=int, default=12345)
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    stem = path.stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"


def image_paths(folder: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def input_paths(path: Path) -> list[Path]:
    suffixes = {".dat", ".jpg", ".jpeg", ".tif", ".tiff"}
    if path.is_file():
        if path.suffix.lower() in suffixes:
            return [path]
        return []
    if path.is_dir():
        dat_paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".dat")
        if dat_paths:
            return dat_paths
        return image_paths(path)
    return []


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad


def percentile_range(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return float(lo), float(hi)


def normalize_uint8(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = percentile_range(values, low, high)
    scaled = (np.asarray(values, dtype=np.float64) - lo) / max(hi - lo, EPS)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def intensity_from_rgb(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    channel_spread = np.max(np.abs(rgb - rgb[:, :, :1]), axis=2)
    is_gray_rgb = bool(np.percentile(channel_spread, 99.9) <= 1.0)
    if is_gray_rgb:
        return rgb[:, :, 0].astype(np.float64, copy=False), "grayscale RGB, red channel used"
    intensity = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2])
    return intensity.astype(np.float64, copy=False), "RGB luminance used"


def _f(value: object, default: float | None = None) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def _i(value: object, default: int | None = None) -> int | None:
    try:
        return int(float(str(value).replace(",", ".")))
    except Exception:
        return default


def parse_dat_header(header_bytes: bytes) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in header_bytes.splitlines():
        if b"=" not in line:
            continue
        key, value = line.split(b"=", 1)
        field = key.decode("ascii", "ignore").split("/")[-1].strip()
        header[field] = value.decode("ascii", "ignore").strip()
    return header


def find_dat_header(header: dict[str, str], hint: str, default: object = None) -> object:
    for key, value in header.items():
        if hint.lower() in key.lower():
            return value
    return default


def dat_dac_bits(header: dict[str, str], default: int = 20) -> int:
    raw = find_dat_header(header, "DAC-Type", None)
    if raw is None:
        return default
    match = re.search(r"\d+", str(raw))
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass
    return default


def trim_dat_stack(stack: np.ndarray) -> tuple[np.ndarray, int]:
    first_channel = stack[0]
    valid_mask = np.isfinite(first_channel) & (first_channel != 0)
    rows, cols = np.where(valid_mask.reshape(first_channel.shape))
    if rows.size:
        last_row = int(rows.max())
        if cols[rows == last_row].max() == first_channel.shape[1] - 1:
            kept_rows = last_row + 1
        else:
            kept_rows = last_row
    else:
        kept_rows = first_channel.shape[0]
    return stack[:, :kept_rows, :], kept_rows


def load_dat_forward_topography(dat_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw = dat_path.read_bytes()
    if b"DATA" not in raw:
        raise ValueError(f"No DATA marker found in {dat_path.name}")
    header_bytes, compressed = raw.split(b"DATA", 1)
    header = parse_dat_header(header_bytes)

    nx = _i(find_dat_header(header, "Num.X", 0), 0)
    ny = _i(find_dat_header(header, "Num.Y", 0), 0)
    if nx is None or ny is None or nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid DAT dimensions in {dat_path.name}")

    if _i(find_dat_header(header, "ScanmodeSine", 0), 0) != 0:
        raise NotImplementedError("Sine scan mode is not supported.")

    payload = zlib.decompress(compressed)
    total_floats = len(payload) // 4
    num_channels = None
    for candidate in (4, 2):
        if total_floats >= candidate * nx * ny:
            num_channels = candidate
            break
    if num_channels is None:
        raise ValueError(f"DAT payload is too small for {dat_path.name}")

    arr = np.frombuffer(payload, dtype="<f4", count=num_channels * ny * nx).copy()
    stack = arr.reshape((num_channels, ny, nx)).astype(np.float32, copy=False)
    stack, kept_rows = trim_dat_stack(stack)

    dac_bits = dat_dac_bits(header, default=20)
    v_per_dac = 10.0 / (2**dac_bits)
    dz_ang_per_dac = _f(find_dat_header(header, "Dacto[A]z", None), None)
    if dz_ang_per_dac is None:
        gain_z = _f(find_dat_header(header, "GainZ", 10.0), 10.0)
        z_piezo = _f(find_dat_header(header, "ZPiezoconst", 19.2), 19.2)
        dz_ang_per_dac = v_per_dac * float(gain_z) * float(z_piezo) * 1.0e2
    z_scale_m_per_dac = float(dz_ang_per_dac) * 1.0e-10

    forward_topography = stack[0].astype(np.float64, copy=False) * z_scale_m_per_dac

    scan_y_down = str(find_dat_header(header, "ScanYDirec", "1")).strip() == "1"
    if scan_y_down:
        forward_topography = np.flip(forward_topography, axis=0)
        # Match dat_to_tiff.py's display orientation for forward channels.
        forward_topography = np.flipud(forward_topography)

    meta = {
        "input_size": [int(nx), int(ny)],
        "kept_rows": int(kept_rows),
        "num_channels_in_dat": int(num_channels),
        "selected_channel": "FT",
        "selected_channel_description": "forward topography only",
        "scan_y_direction_raw": str(find_dat_header(header, "ScanYDirec", "1")).strip(),
        "dac_bits": float(dac_bits),
        "v_per_dac": float(v_per_dac),
        "z_scale_m_per_dac": float(z_scale_m_per_dac),
    }
    return forward_topography.astype(np.float64, copy=False), meta


def robust_affine_normalization(values: np.ndarray) -> tuple[float, float, dict[str, float | str]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0, {
            "method": "fallback_no_finite_values",
            "robust_low_percentile": 1.0,
            "robust_high_percentile": 99.0,
        }

    low_pct = 1.0
    high_pct = 99.0
    low, high = np.percentile(finite, [low_pct, high_pct])
    span = float(high - low)
    method = "p1_p99_affine_no_clipping"

    if not np.isfinite(span) or span <= EPS:
        low_pct = 0.1
        high_pct = 99.9
        low, high = np.percentile(finite, [low_pct, high_pct])
        span = float(high - low)
        method = "p0.1_p99.9_affine_no_clipping"

    if not np.isfinite(span) or span <= EPS:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        span = float(high - low)
        method = "min_max_affine_no_clipping"

    if not np.isfinite(span) or span <= EPS:
        low = float(np.nanmedian(finite))
        span = 1.0
        method = "constant_image_fallback"

    return float(low), float(span), {
        "method": method,
        "robust_low_percentile": float(low_pct),
        "robust_high_percentile": float(high_pct),
        "native_min": float(np.nanmin(finite)),
        "native_max": float(np.nanmax(finite)),
        "native_median": float(np.nanmedian(finite)),
    }


def read_height_proxy(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = path.suffix.lower()
    dat_meta: dict[str, Any] | None = None
    if suffix == ".dat":
        intensity, dat_meta = load_dat_forward_topography(path)
        mode = "DAT"
        size = [int(intensity.shape[1]), int(intensity.shape[0])]
        conversion = "DAT forward topography channel scaled to native metres"
        normalization_offset, normalization_scale, normalization_stats = robust_affine_normalization(intensity)
        z = (intensity - normalization_offset) / normalization_scale
        height_proxy = (
            "DAT forward topography native values robustly normalized as "
            "(value - normalization_offset) / normalization_scale"
        )
        normalized_to_unit_interval = False
        native_outputs_rescaled = True
        scale_reapply_note = (
            "Native-scale outputs are recovered by multiplying normalized values by "
            "normalization_scale and adding normalization_offset where appropriate. "
            "Removed backgrounds are multiplied by normalization_scale only."
        )
    else:
        with Image.open(path) as img:
            mode = img.mode
            size = list(img.size)

            if suffix in {".tif", ".tiff"}:
                raw = np.asarray(img)
                if raw.ndim == 2:
                    intensity = raw.astype(np.float64, copy=False)
                    conversion = f"TIFF grayscale ({raw.dtype}) used"
                elif raw.ndim == 3 and raw.shape[2] >= 3:
                    rgb = raw[:, :, :3].astype(np.float64, copy=False)
                    intensity, conversion = intensity_from_rgb(rgb)
                else:
                    rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
                    intensity, conversion = intensity_from_rgb(rgb)

                normalization_offset, normalization_scale, normalization_stats = robust_affine_normalization(intensity)
                z = (intensity - normalization_offset) / normalization_scale
                height_proxy = (
                    "TIFF native values robustly normalized as "
                    "(value - normalization_offset) / normalization_scale"
                )
                normalized_to_unit_interval = False
                native_outputs_rescaled = True
                scale_reapply_note = (
                    "Native-scale outputs are recovered by multiplying normalized values by "
                    "normalization_scale and adding normalization_offset where appropriate. "
                    "Removed backgrounds are multiplied by normalization_scale only."
                )
            else:
                rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
                intensity, conversion = intensity_from_rgb(rgb)
                normalization_offset = 0.0
                normalization_scale = 255.0
                normalization_stats = {
                    "method": "jpeg_8bit_divide_by_255",
                    "robust_low_percentile": None,
                    "robust_high_percentile": None,
                }
                z = intensity / normalization_scale
                height_proxy = "JPEG intensity normalized to [0, 1]"
                normalized_to_unit_interval = True
                native_outputs_rescaled = True
                scale_reapply_note = "Native-scale outputs are recovered by multiplying by normalization_scale."

    if suffix in {".jpg", ".jpeg"}:
        note = "Physical height calibration is unavailable from the JPEG alone."
    elif suffix == ".dat":
        note = "Forward topography was parsed from the DAT file and saved outputs preserve native metre scaling."
    else:
        note = "TIFF native numeric units are preserved for saved array outputs."
    meta = {
        "input_mode": mode,
        "input_size": list(size),
        "input_suffix": suffix,
        "height_proxy": height_proxy,
        "conversion": conversion,
        "normalization_offset": float(normalization_offset),
        "normalization_scale": float(normalization_scale),
        "normalization_factor": float(normalization_scale),
        "normalization": normalization_stats,
        "normalized_to_unit_interval": normalized_to_unit_interval,
        "native_outputs_rescaled_by_factor": native_outputs_rescaled,
        "scale_reapply_note": scale_reapply_note,
        "input_note": note,
    }
    if dat_meta is not None:
        meta["dat"] = dat_meta
    return z, meta


def make_labels(fit_mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[int]]:
    min_area = max(
        int(args.min_component_pixels),
        int(round(args.min_component_fraction * fit_mask.size)),
    )
    structure = np.ones((3, 3), dtype=bool)
    raw_labels, n_labels = ndi.label(fit_mask, structure=structure)
    if n_labels == 0:
        return np.zeros(fit_mask.shape, dtype=np.int32), []

    areas = np.bincount(raw_labels.ravel())
    kept = [idx for idx in range(1, len(areas)) if areas[idx] >= min_area]
    kept.sort(key=lambda idx: int(areas[idx]), reverse=True)
    kept = kept[: int(args.max_components)]

    labels = np.zeros_like(raw_labels, dtype=np.int32)
    component_areas: list[int] = []
    for new_label, old_label in enumerate(kept, start=1):
        labels[raw_labels == old_label] = new_label
        component_areas.append(int(areas[old_label]))
    return labels, component_areas

def crop_trailing_blank_rows(
    z: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    # Treat trailing rows that are entirely 0, 1.0, or non-finite as blank padding.
    row_is_blank = np.all((~np.isfinite(z)) | (z == 0) | (z == 1.0), axis=1)

    if np.all(row_is_blank):
        return z[:0, :], {"bottom": z.shape[0], "kept_rows": 0}

    last_data_row = int(np.where(~row_is_blank)[0][-1])

    crop_end = last_data_row + 1
    cropped = z[:crop_end, :]

    meta = {
        "bottom": int(z.shape[0] - crop_end),
        "kept_rows": crop_end,
    }
    return cropped, meta


def build_masks(z: np.ndarray, args: argparse.Namespace) -> MaskResult:
    finite = np.isfinite(z)
    warnings: list[str] = []
    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")

    smoothed = ndi.gaussian_filter(np.where(finite, z, np.nanmedian(z[finite])), sigma=args.gradient_sigma)
    grad = sobel(smoothed)
    grad_values = grad[finite]
    grad_med = float(np.median(grad_values))
    grad_sigma = robust_sigma(grad_values)
    edge_threshold_percentile = float(np.percentile(grad_values, args.edge_percentile))
    edge_threshold_mad = grad_med + args.edge_mad_multiplier * grad_sigma
    edge_threshold = max(edge_threshold_percentile, edge_threshold_mad)
    raw_edge_mask = finite & (grad >= edge_threshold)

    local = ndi.median_filter(np.where(finite, z, np.nanmedian(z[finite])), size=args.defect_filter_size)
    local_residual = np.abs(z - local)
    residual_values = local_residual[finite]
    residual_med = float(np.median(residual_values))
    residual_sigma = robust_sigma(residual_values)
    defect_threshold_percentile = float(np.percentile(residual_values, args.defect_percentile))
    defect_threshold_mad = residual_med + args.defect_mad_multiplier * residual_sigma
    defect_threshold = max(defect_threshold_percentile, defect_threshold_mad)
    raw_defect_mask = finite & (local_residual >= defect_threshold)

    edge_mask = raw_edge_mask
    if args.edge_dilate_px > 0:
        edge_mask = binary_dilation(edge_mask, footprint=disk(args.edge_dilate_px))
    defect_mask = raw_defect_mask
    if args.defect_dilate_px > 0:
        defect_mask = binary_dilation(defect_mask, footprint=disk(args.defect_dilate_px))

    feature_mask = finite & (edge_mask | defect_mask)
    fit_mask = finite & ~feature_mask
    labels, component_areas = make_labels(fit_mask, args)

    fit_fraction = float(np.count_nonzero(labels) / max(np.count_nonzero(finite), 1))
    if fit_fraction < args.min_fit_fraction:
        warnings.append(
            "Few usable fit pixels after edge/defect masking; retrying with edge mask only."
        )
        feature_mask = finite & edge_mask
        fit_mask = finite & ~feature_mask
        labels, component_areas = make_labels(fit_mask, args)
        fit_fraction = float(np.count_nonzero(labels) / max(np.count_nonzero(finite), 1))

    if fit_fraction < args.min_fit_fraction:
        warnings.append(
            "Fit mask is still sparse; falling back to one connected finite region. "
            "Step-edge protection is weaker for this image."
        )
        labels = np.zeros(z.shape, dtype=np.int32)
        labels[finite] = 1
        component_areas = [int(np.count_nonzero(finite))]

    thresholds = {
        "gradient_median": grad_med,
        "gradient_sigma_mad": grad_sigma,
        "edge_threshold_percentile": edge_threshold_percentile,
        "edge_threshold_mad": edge_threshold_mad,
        "edge_threshold_used": float(edge_threshold),
        "local_residual_median": residual_med,
        "local_residual_sigma_mad": residual_sigma,
        "defect_threshold_percentile": defect_threshold_percentile,
        "defect_threshold_mad": defect_threshold_mad,
        "defect_threshold_used": float(defect_threshold),
        "fit_fraction": fit_fraction,
    }
    return MaskResult(
        feature_mask=feature_mask.astype(bool),
        edge_mask=edge_mask.astype(bool),
        defect_mask=defect_mask.astype(bool),
        labels=labels.astype(np.int32, copy=False),
        thresholds=thresholds,
        component_areas=component_areas,
        warnings=warnings,
    )


def normalized_xy(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    y = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, cols, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def poly_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    if degree <= 0:
        return np.empty((x.size, 0), dtype=np.float64)
    cols = [x.ravel(), y.ravel()]
    if degree >= 2:
        cols.extend([(x * x).ravel(), (x * y).ravel(), (y * y).ravel()])
    return np.column_stack(cols).astype(np.float64, copy=False)


def full_background(shape: tuple[int, int], degree: int, beta_poly: np.ndarray) -> np.ndarray:
    if degree <= 0 or beta_poly.size == 0:
        return np.zeros(shape, dtype=np.float64)
    xx, yy = normalized_xy(shape)
    terms = poly_terms(xx, yy, degree)
    return (terms @ beta_poly).reshape(shape)


def candidate_score(z: np.ndarray, background: np.ndarray, labels: np.ndarray, n_poly_terms: int) -> tuple[float, float, float]:
    mask = (labels > 0) & np.isfinite(z) & np.isfinite(background)
    if not np.any(mask):
        return math.inf, math.inf, math.inf

    corrected = z[mask] - background[mask]
    label_values = labels[mask]
    residual = np.empty_like(corrected, dtype=np.float64)
    for label in np.unique(label_values):
        idx = label_values == label
        residual[idx] = corrected[idx] - np.median(corrected[idx])

    score = robust_sigma(residual)
    p95_abs = float(np.percentile(np.abs(residual[np.isfinite(residual)]), 95.0))
    n = max(int(residual.size), 1)
    bic_like = n * math.log(score * score + EPS) + n_poly_terms * math.log(n)
    return float(score), p95_abs, float(bic_like)


def sample_fit_pixels(labels: np.ndarray, max_fit_pixels: int, rng: np.random.Generator) -> np.ndarray:
    flat_indices = np.flatnonzero(labels.ravel() > 0)
    if flat_indices.size <= max_fit_pixels:
        return flat_indices

    label_flat = labels.ravel()
    selected_parts: list[np.ndarray] = []
    remaining_budget = int(max_fit_pixels)
    unique_labels = np.unique(label_flat[flat_indices])

    # Keep representation from every retained component before random fill.
    per_label_floor = max(20, min(300, max_fit_pixels // max(len(unique_labels), 1) // 2))
    for label in unique_labels:
        idx = flat_indices[label_flat[flat_indices] == label]
        take = min(idx.size, per_label_floor, remaining_budget)
        if take > 0:
            selected_parts.append(rng.choice(idx, size=take, replace=False))
            remaining_budget -= take
        if remaining_budget <= 0:
            break

    if remaining_budget > 0:
        already = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
        already_set = set(int(i) for i in already)
        rest = np.array([i for i in flat_indices if int(i) not in already_set], dtype=np.int64)
        if rest.size > 0:
            selected_parts.append(rng.choice(rest, size=min(rest.size, remaining_budget), replace=False))

    selected = np.unique(np.concatenate(selected_parts))
    return selected.astype(np.int64, copy=False)


def solve_candidate(
    z: np.ndarray,
    labels: np.ndarray,
    degree: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Candidate:
    name = {0: "none", 1: "plane", 2: "quadratic"}.get(degree, f"degree_{degree}")
    n_poly_terms = 0 if degree == 0 else (2 if degree == 1 else 5)
    if degree == 0:
        background = np.zeros_like(z, dtype=np.float64)
        score, p95_abs, bic_like = candidate_score(z, background, labels, n_poly_terms)
        return Candidate(
            name=name,
            degree=degree,
            background=background,
            score_mad=score,
            residual_p95_abs=p95_abs,
            background_range_98=0.0,
            n_poly_terms=n_poly_terms,
            fit_pixels=int(np.count_nonzero(labels)),
            n_regions=int(labels.max()),
            bic_like=bic_like,
        )

    fit_indices = sample_fit_pixels(labels, int(args.max_fit_pixels), rng)
    if fit_indices.size < max(20, n_poly_terms + int(labels.max()) + 1):
        background = np.zeros_like(z, dtype=np.float64)
        score, p95_abs, bic_like = candidate_score(z, background, labels, n_poly_terms)
        return Candidate(
            name=name,
            degree=degree,
            background=background,
            score_mad=score,
            residual_p95_abs=p95_abs,
            background_range_98=0.0,
            n_poly_terms=n_poly_terms,
            fit_pixels=int(fit_indices.size),
            n_regions=int(labels.max()),
            bic_like=bic_like,
            success=False,
            message="Not enough fit pixels for polynomial plus component-offset model.",
        )

    rows, cols = z.shape
    yy_i, xx_i = np.unravel_index(fit_indices, z.shape)
    x = -1.0 + 2.0 * xx_i.astype(np.float64) / max(cols - 1, 1)
    y = -1.0 + 2.0 * yy_i.astype(np.float64) / max(rows - 1, 1)
    z_fit = z.ravel()[fit_indices].astype(np.float64, copy=False)
    sample_labels = labels.ravel()[fit_indices]

    unique_labels, inverse_labels = np.unique(sample_labels, return_inverse=True)
    n_regions = int(unique_labels.size)
    poly = poly_terms(x, y, degree)
    offset = csr_matrix(
        (
            np.ones(fit_indices.size, dtype=np.float64),
            (np.arange(fit_indices.size), inverse_labels),
        ),
        shape=(fit_indices.size, n_regions),
    )
    design = hstack([csr_matrix(poly), offset], format="csr")

    weights = np.ones(fit_indices.size, dtype=np.float64)
    beta = np.zeros(design.shape[1], dtype=np.float64)
    lsqr_result = None
    for _ in range(8):
        sqrt_w = np.sqrt(np.clip(weights, 0.0, np.inf))
        weighted_design = design.multiply(sqrt_w[:, None])
        weighted_z = z_fit * sqrt_w
        lsqr_result = lsqr(
            weighted_design,
            weighted_z,
            atol=1.0e-10,
            btol=1.0e-10,
            iter_lim=max(500, 4 * design.shape[1]),
        )
        beta = lsqr_result[0]
        residual = z_fit - design @ beta
        sigma = robust_sigma(residual)
        if sigma <= EPS:
            break
        cutoff = 1.5 * sigma
        new_weights = np.minimum(1.0, cutoff / (np.abs(residual) + EPS))
        if np.max(np.abs(new_weights - weights)) < 0.01:
            weights = new_weights
            break
        weights = new_weights

    beta_poly = beta[:n_poly_terms]
    background = full_background(z.shape, degree, beta_poly)
    score, p95_abs, bic_like = candidate_score(z, background, labels, n_poly_terms)
    bg_lo, bg_hi = percentile_range(background[np.isfinite(z)], 1.0, 99.0)
    condition = None
    iterations = None
    if lsqr_result is not None:
        iterations = int(lsqr_result[2])
        condition = float(lsqr_result[6])

    return Candidate(
        name=name,
        degree=degree,
        background=background,
        score_mad=score,
        residual_p95_abs=p95_abs,
        background_range_98=float(bg_hi - bg_lo),
        n_poly_terms=n_poly_terms,
        fit_pixels=int(fit_indices.size),
        n_regions=n_regions,
        bic_like=bic_like,
        solver_iterations=iterations,
        solver_condition_estimate=condition,
    )


def background_delta_range(a: np.ndarray, b: np.ndarray) -> float:
    delta = a - b
    lo, hi = percentile_range(delta, 1.0, 99.0)
    return float(hi - lo)


def selection_metric(candidate: Candidate, use_p95: bool) -> float:
    return candidate.residual_p95_abs if use_p95 else candidate.score_mad


def select_candidate(candidates: list[Candidate], args: argparse.Namespace) -> tuple[Candidate, str]:
    by_degree = {c.degree: c for c in candidates if c.success}
    none = by_degree[0]
    plane = by_degree.get(1)
    quadratic = by_degree.get(2)
    selected = none
    use_p95 = none.score_mad <= EPS
    metric_name = "95th-percentile absolute residual" if use_p95 else "robust MAD residual"
    reason = f"No correction: plane/quadratic did not pass the improvement thresholds for {metric_name}."

    def enough_background(c: Candidate, reference_score: float) -> bool:
        threshold = max(args.min_background_range, 0.5 * reference_score)
        return c.background_range_98 >= threshold

    none_score = selection_metric(none, use_p95)

    if plane is not None and np.isfinite(selection_metric(plane, use_p95)) and np.isfinite(none_score):
        plane_score = selection_metric(plane, use_p95)
        abs_gain = none_score - plane_score
        required = max(args.min_abs_improvement, args.plane_min_relative_improvement * none_score)
        if abs_gain >= required and enough_background(plane, none_score):
            selected = plane
            rel_gain = 100.0 * abs_gain / max(none_score, EPS)
            reason = (
                f"Plane correction: {metric_name} improved by {rel_gain:.1f}% over no correction."
            )

    if quadratic is not None and np.isfinite(selection_metric(quadratic, use_p95)):
        if selected.degree == 1 and plane is not None:
            plane_score = selection_metric(plane, use_p95)
            quadratic_score = selection_metric(quadratic, use_p95)
            abs_gain = plane_score - quadratic_score
            required = max(args.min_abs_improvement, args.quadratic_min_relative_improvement * plane_score)
            extra_range = background_delta_range(quadratic.background, plane.background)
            extra_required = max(args.min_background_range, 0.35 * plane_score)
            if abs_gain >= required and extra_range >= extra_required:
                selected = quadratic
                rel_gain = 100.0 * abs_gain / max(plane_score, EPS)
                reason = (
                    f"Quadratic correction: {metric_name} improved by {rel_gain:.1f}% beyond plane, "
                    "and the added curvature exceeded the background-range threshold."
                )
        elif selected.degree == 0:
            quadratic_score = selection_metric(quadratic, use_p95)
            abs_gain = none_score - quadratic_score
            required = max(
                args.min_abs_improvement,
                args.quadratic_from_none_min_relative_improvement * none_score,
            )
            if abs_gain >= required and enough_background(quadratic, none_score):
                selected = quadratic
                rel_gain = 100.0 * abs_gain / max(none_score, EPS)
                reason = (
                    f"Quadratic correction: plane was not sufficient, but quadratic improved "
                    f"the {metric_name} by {rel_gain:.1f}% over no correction."
                )

    return selected, reason


def labels_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    max_label = int(labels.max())
    if max_label <= 0:
        return rgb
    for label in range(1, max_label + 1):
        hue = (label * 0.61803398875) % 1.0
        rgb[labels == label] = hsv_to_rgb(hue, 0.75, 0.95)
    return rgb


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0))


def mask_overlay(z: np.ndarray, mask_result: MaskResult) -> np.ndarray:
    base = normalize_uint8(z)
    rgb = np.repeat(base[:, :, None], 3, axis=2).astype(np.float64)
    region_rgb = labels_rgb(mask_result.labels).astype(np.float64)
    region_mask = mask_result.labels > 0
    rgb[region_mask] = 0.55 * rgb[region_mask] + 0.45 * region_rgb[region_mask]
    rgb[mask_result.feature_mask] = [255.0, 45.0, 35.0]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def residual_map(z: np.ndarray, background: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.full(z.shape, np.nan, dtype=np.float64)
    mask = (labels > 0) & np.isfinite(z) & np.isfinite(background)
    corrected = z - background
    for label in np.unique(labels[mask]):
        idx = labels == label
        idx &= mask
        if np.any(idx):
            result[idx] = corrected[idx] - np.median(corrected[idx])
    return result


def diagnostic_display_limits(values: np.ndarray, robust_clip: bool) -> tuple[float | None, float | None]:
    if not robust_clip:
        return None, None
    vmin, vmax = percentile_range(values, 1.0, 99.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None, None
    return vmin, vmax


def save_diagnostics(
    z: np.ndarray,
    flattened: np.ndarray,
    selected: Candidate,
    candidates: list[Candidate],
    mask_result: MaskResult,
    output_prefix: Path,
    output_scale: float,
    output_offset: float,
    robust_gray_display: bool,
    original_title: str,
    write_flattened_tiff: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    paths = {
        "flattened_png": str(output_prefix.with_name(output_prefix.name + "_flattened.png")),
        "background_png": str(output_prefix.with_name(output_prefix.name + "_background.png")),
        "fit_mask_png": str(output_prefix.with_name(output_prefix.name + "_fit_mask.png")),
        "diagnostic_png": str(output_prefix.with_name(output_prefix.name + "_diagnostic.png")),
        "flattened_npy": str(output_prefix.with_name(output_prefix.name + "_flattened.npy")),
    }

    z_native = z * output_scale + output_offset
    flattened_native = flattened * output_scale + output_offset
    background_native = selected.background * output_scale

    Image.fromarray(normalize_uint8(flattened_native)).save(paths["flattened_png"])
    Image.fromarray(normalize_uint8(background_native)).save(paths["background_png"])
    Image.fromarray(mask_overlay(z_native, mask_result)).save(paths["fit_mask_png"])
    np.save(paths["flattened_npy"], flattened_native.astype(np.float32))
    if write_flattened_tiff:
        paths["flattened_tiff"] = str(output_prefix.with_name(output_prefix.name + "_flattened.tiff"))
        tifffile.imwrite(
            paths["flattened_tiff"],
            flattened_native.astype(np.float32, copy=False),
            dtype=np.float32,
        )

    residual = residual_map(z_native, background_native, mask_result.labels)
    original_vmin, original_vmax = diagnostic_display_limits(z_native, robust_gray_display)
    flattened_vmin, flattened_vmax = diagnostic_display_limits(flattened_native, robust_gray_display)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    axes = axes.ravel()

    axes[0].imshow(z_native, cmap="gray", vmin=original_vmin, vmax=original_vmax)
    title_suffix = " (p1-p99 display)" if robust_gray_display else ""
    axes[0].set_title(f"{original_title}{title_suffix}")

    axes[1].imshow(mask_overlay(z_native, mask_result))
    axes[1].set_title("Fit regions; red excluded")

    bg = background_native
    bg_abs = max(abs(float(np.nanpercentile(bg, 1))), abs(float(np.nanpercentile(bg, 99))), EPS)
    axes[2].imshow(bg, cmap="coolwarm", vmin=-bg_abs, vmax=bg_abs)
    axes[2].set_title(f"Removed background: {selected.name}")

    axes[3].imshow(flattened_native, cmap="gray", vmin=flattened_vmin, vmax=flattened_vmax)
    axes[3].set_title(f"Flattened intensity{title_suffix}")

    axes[4].imshow(residual, cmap="coolwarm")
    axes[4].set_title("Within-region residual")

    names = [c.name for c in candidates]
    scores = [c.score_mad for c in candidates]
    colors = ["#808080" if c.degree != selected.degree else "#207a3a" for c in candidates]
    axes[5].bar(names, scores, color=colors)
    axes[5].set_title("Robust residual score")
    axes[5].set_ylabel("MAD sigma, lower is better")
    axes[5].tick_params(axis="x", rotation=20)

    for ax in axes[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(paths["diagnostic_png"], dpi=160)
    plt.close(fig)
    display_meta = {
        "robust_gray_display_for_diagnostic_png": robust_gray_display,
        "display_clipping_note": (
            "Only the diagnostic PNG grayscale panels are clipped for visibility; "
            "fitting and numeric outputs use the unclipped data."
        ),
        "clip_percentiles": [1.0, 99.0] if robust_gray_display else None,
        "original_display_vmin_native_units": original_vmin,
        "original_display_vmax_native_units": original_vmax,
        "flattened_display_vmin_native_units": flattened_vmin,
        "flattened_display_vmax_native_units": flattened_vmax,
    }
    return paths, display_meta


def candidate_to_report(candidate: Candidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "degree": candidate.degree,
        "success": candidate.success,
        "message": candidate.message,
        "score_mad_sigma": candidate.score_mad,
        "residual_p95_abs": candidate.residual_p95_abs,
        "background_range_1_to_99": candidate.background_range_98,
        "n_poly_terms": candidate.n_poly_terms,
        "fit_pixels": candidate.fit_pixels,
        "n_regions_in_fit": candidate.n_regions,
        "bic_like": candidate.bic_like,
        "solver_iterations": candidate.solver_iterations,
        "solver_condition_estimate": candidate.solver_condition_estimate,
    }


def process_image(path: Path, output_folder: Path, args: argparse.Namespace) -> dict[str, Any]:
    z, input_meta = read_height_proxy(path)
    output_scale = float(input_meta.get("normalization_scale", input_meta.get("normalization_factor", 1.0)))
    output_offset = float(input_meta.get("normalization_offset", 0.0))
    suffix = path.suffix.lower()
    is_native_height = suffix in {".dat", ".tif", ".tiff"}
    if suffix == ".dat":
        original_title = "Original DAT forward topography"
    elif suffix in {".tif", ".tiff"}:
        original_title = "Original TIFF native values"
    else:
        original_title = "Original JPEG intensity"

    z, crop_meta = crop_trailing_blank_rows(z)
    input_meta["crop"] = crop_meta

    mask_result = build_masks(z, args)
    rng = np.random.default_rng(args.random_seed)

    candidates = [
        solve_candidate(z, mask_result.labels, 0, args, rng),
        solve_candidate(z, mask_result.labels, 1, args, rng),
        solve_candidate(z, mask_result.labels, 2, args, rng),
    ]
    selected, reason = select_candidate(candidates, args)
    flattened = z - selected.background

    output_prefix = output_folder / safe_stem(path)
    output_paths, diagnostic_display = save_diagnostics(
        z,
        flattened,
        selected,
        candidates,
        mask_result,
        output_prefix,
        output_scale,
        output_offset,
        robust_gray_display=is_native_height,
        original_title=original_title,
        write_flattened_tiff=is_native_height,
    )
    report_path = output_prefix.with_name(output_prefix.name + "_report.json")

    report: dict[str, Any] = {
        "input_path": str(path),
        "input": input_meta,
        "image_shape_rows_cols": list(z.shape),
        "selected_correction": selected.name,
        "selected_degree": selected.degree,
        "selection_reason": reason,
        "scientific_safeguards": [
            "High-gradient step edges and sharp defects were excluded from the fit.",
            "Connected smooth regions were given independent constant offsets during fitting.",
            "Only the polynomial background was subtracted; region offsets were preserved.",
            "No/plane/quadratic candidates were compared with a preference for the simplest sufficient model.",
        ],
        "masking": {
            "edge_pixels": int(np.count_nonzero(mask_result.edge_mask)),
            "defect_pixels": int(np.count_nonzero(mask_result.defect_mask)),
            "excluded_pixels": int(np.count_nonzero(mask_result.feature_mask)),
            "fit_pixels": int(np.count_nonzero(mask_result.labels)),
            "n_fit_regions": int(mask_result.labels.max()),
            "component_areas_pixels": mask_result.component_areas,
            "thresholds": mask_result.thresholds,
            "warnings": mask_result.warnings,
        },
        "candidates": [candidate_to_report(c) for c in candidates],
        "outputs": output_paths,
        "diagnostic_display": diagnostic_display,
        "scientific_data_output": {
            "path": output_paths.get("flattened_tiff", output_paths["flattened_npy"]),
            "format": "float32_tiff_native_units" if is_native_height else "float32_npy_intensity_units",
            "scaling_note": (
                "For DAT/TIFF inputs this is unclipped flattened data in the same native numeric units "
                "as the parsed input topography. It is computed as native_input - native_background."
                if is_native_height
                else "For JPEG inputs this remains the normalized intensity-derived height proxy."
            ),
        },
        "output_normalization_scale": output_scale,
        "output_normalization_offset": output_offset,
    }
    report["outputs"]["report_json"] = str(report_path)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    paths = input_paths(input_path)
    if not paths:
        raise SystemExit(f"No .dat/.tif/.tiff/.jpg/.jpeg files found in {input_path}")

    summary: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            report = process_image(path, output_folder, args)
            summary.append(
                {
                    "input": str(path),
                    "selected_correction": report["selected_correction"],
                    "selection_reason": report["selection_reason"],
                    "report_json": report["outputs"]["report_json"],
                }
            )
            print(f"[OK] {path.name}: {report['selected_correction']} - {report['selection_reason']}")
        except Exception as exc:  # Continue batch processing other images.
            failures.append({"input": str(path), "error": repr(exc)})
            print(f"[FAIL] {path.name}: {exc}")

    batch_report = {
        "input_path": str(input_path),
        "output_folder": str(output_folder),
        "n_images": len(paths),
        "n_success": len(summary),
        "n_failures": len(failures),
        "images": summary,
        "failures": failures,
    }
    with (output_folder / "batch_report.json").open("w", encoding="utf-8") as f:
        json.dump(batch_report, f, indent=2)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
