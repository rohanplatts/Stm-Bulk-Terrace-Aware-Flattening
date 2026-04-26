#!/usr/bin/env python3
"""
Generate demo comparison PNGs for all DAT files in a folder.

Each output image contains four side-by-side panels:
  1. Human-display forward-topography image.
  2. Naive global plane-flattened image.
  3. Naive global quadratic-flattened image.
  4. The masked stm_flatten.py result.

The two baseline fits deliberately use all finite pixels without masking step
edges. This demonstrates why the production method is needed for terrace data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stm_flatten


PANEL_TITLES = (
    "FT display",
    "Global plane",
    "Global quadratic",
    "stm_flatten",
)


def default_flatten_args() -> argparse.Namespace:
    return argparse.Namespace(
        edge_percentile=90.0,
        edge_mad_multiplier=5.0,
        edge_dilate_px=7,
        gradient_sigma=1.0,
        defect_percentile=98.5,
        defect_mad_multiplier=8.0,
        defect_filter_size=5,
        defect_dilate_px=1,
        min_component_pixels=160,
        min_component_fraction=0.0025,
        max_components=80,
        min_fit_fraction=0.08,
        max_fit_pixels=120000,
        plane_min_relative_improvement=0.05,
        quadratic_min_relative_improvement=0.08,
        quadratic_from_none_min_relative_improvement=0.12,
        min_abs_improvement=0.002,
        min_background_range=0.005,
        random_seed=12345,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build side-by-side flattening demo PNGs for DAT files.")
    parser.add_argument("--input_dir", default="in", help="Folder containing .dat files.")
    parser.add_argument("--output_dir", default="demo", help="Folder for demo PNG outputs.")
    parser.add_argument("--tile_size", type=int, default=360, help="Displayed size of each panel in pixels.")
    return parser.parse_args()


def polynomial_terms(shape: tuple[int, int], degree: int, mask: np.ndarray) -> np.ndarray:
    rows, cols = shape
    yy, xx = np.where(mask)
    x = -1.0 + 2.0 * xx.astype(np.float64) / max(cols - 1, 1)
    y = -1.0 + 2.0 * yy.astype(np.float64) / max(rows - 1, 1)
    terms = [np.ones_like(x), x, y]
    if degree >= 2:
        terms.extend([x * x, x * y, y * y])
    return np.column_stack(terms)


def full_polynomial_background(shape: tuple[int, int], degree: int, beta: np.ndarray) -> np.ndarray:
    rows, cols = shape
    y_axis = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    x_axis = np.linspace(-1.0, 1.0, cols, dtype=np.float64)
    x, y = np.meshgrid(x_axis, y_axis)
    terms = [np.ones(x.size, dtype=np.float64), x.ravel(), y.ravel()]
    if degree >= 2:
        terms.extend([(x * x).ravel(), (x * y).ravel(), (y * y).ravel()])
    design = np.column_stack(terms)
    return (design @ beta).reshape(shape)


def global_polynomial_flatten(native: np.ndarray, degree: int) -> np.ndarray:
    mask = np.isfinite(native)
    if np.count_nonzero(mask) < (3 if degree == 1 else 6):
        return native.copy()
    design = polynomial_terms(native.shape, degree, mask)
    values = native[mask].astype(np.float64, copy=False)
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    background = full_polynomial_background(native.shape, degree, beta)
    return native - background


def stm_flatten_native(dat_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    z, meta = stm_flatten.read_height_proxy(dat_path)
    z, _crop_meta = stm_flatten.crop_trailing_blank_rows(z)
    mask_result = stm_flatten.build_masks(z, args)
    rng = np.random.default_rng(args.random_seed)
    candidates = [
        stm_flatten.solve_candidate(z, mask_result.labels, 0, args, rng),
        stm_flatten.solve_candidate(z, mask_result.labels, 1, args, rng),
        stm_flatten.solve_candidate(z, mask_result.labels, 2, args, rng),
    ]
    selected, _reason = stm_flatten.select_candidate(candidates, args)
    output_scale = float(meta.get("normalization_scale", meta.get("normalization_factor", 1.0)))
    output_offset = float(meta.get("normalization_offset", 0.0))
    flattened = z - selected.background
    return flattened * output_scale + output_offset, selected.name


def clipped_rgb(values: np.ndarray) -> Image.Image:
    return Image.fromarray(stm_flatten.normalize_uint8(values), mode="L").convert("RGB")


def load_display_inputs(dat_path: Path) -> np.ndarray:
    z, meta = stm_flatten.read_height_proxy(dat_path)
    z, _crop_meta = stm_flatten.crop_trailing_blank_rows(z)
    output_scale = float(meta.get("normalization_scale", meta.get("normalization_factor", 1.0)))
    output_offset = float(meta.get("normalization_offset", 0.0))
    return z * output_scale + output_offset


def fit_panel(image: Image.Image, tile_size: int) -> Image.Image:
    panel = Image.new("RGB", (tile_size, tile_size), "white")
    image = image.copy()
    image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    x = (tile_size - image.width) // 2
    y = (tile_size - image.height) // 2
    panel.paste(image, (x, y))
    return panel


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_comparison_png(
    dat_path: Path,
    output_path: Path,
    panels: list[Image.Image],
    stm_model: str,
    tile_size: int,
) -> None:
    title_h = 46
    header_h = 30
    margin = 12
    gap = 8
    width = 2 * margin + 4 * tile_size + 3 * gap
    height = margin + title_h + header_h + tile_size + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = font(18, bold=True)
    label_font = font(13, bold=True)
    draw.text((margin, margin), dat_path.name, fill=(20, 20, 20), font=title_font)
    draw.text(
        (margin, margin + 24),
        "Naive global fits use all pixels; stm_flatten masks step edges/defects and preserves terrace offsets.",
        fill=(90, 90, 90),
        font=font(11),
    )

    y_label = margin + title_h
    y_panel = y_label + header_h
    for idx, panel in enumerate(panels):
        x = margin + idx * (tile_size + gap)
        label = PANEL_TITLES[idx]
        if idx == 3:
            label = f"{label} ({stm_model})"
        draw.rectangle((x, y_label, x + tile_size, y_label + header_h - 4), fill=(33, 37, 41))
        draw.text((x + 8, y_label + 6), label, fill="white", font=label_font)
        canvas.paste(fit_panel(panel, tile_size), (x, y_panel))
        draw.rectangle((x, y_panel, x + tile_size, y_panel + tile_size), outline=(150, 150, 150))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def process_one(dat_path: Path, output_dir: Path, args: argparse.Namespace, tile_size: int) -> Path:
    native = load_display_inputs(dat_path)
    plane = global_polynomial_flatten(native, degree=1)
    quadratic = global_polynomial_flatten(native, degree=2)
    flattened, stm_model = stm_flatten_native(dat_path, args)

    panels = [
        clipped_rgb(native),
        clipped_rgb(plane),
        clipped_rgb(quadratic),
        clipped_rgb(flattened),
    ]
    out_path = output_dir / f"{stm_flatten.safe_stem(dat_path)}_demo.png"
    save_comparison_png(dat_path, out_path, panels, stm_model, tile_size)
    return out_path


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dat_paths = sorted(input_dir.glob("*.dat"))
    if not dat_paths:
        raise SystemExit(f"No .dat files found in {input_dir}")

    flatten_args = default_flatten_args()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, dat_path in enumerate(dat_paths, start=1):
        out_path = process_one(dat_path, output_dir, flatten_args, args.tile_size)
        print(f"[{index}/{len(dat_paths)}] {dat_path.name} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
