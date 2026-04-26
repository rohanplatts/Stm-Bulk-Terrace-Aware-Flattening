# How `stm_flatten` Works

This document describes the scientific procedure implemented by `stm_flatten.py`.

## Goal

The goal is to remove slow scan/background artefacts while preserving real STM
terraces and step heights. The algorithm is conservative by design: it compares
no correction, plane correction, and quadratic correction, then chooses the
simplest correction that produces a meaningful improvement.

## Inputs

Supported inputs are:

- `.dat`: parsed directly. Only the forward topography channel is used.
- `.tif` / `.tiff`: read as native numeric image data.
- `.jpg` / `.jpeg`: read as an 8-bit intensity height proxy.

For folders, `.dat` files take priority. If a folder contains both `.dat` files
and converted TIFF/JPEG exports, only the `.dat` files are processed by default.

## DAT Parsing

For DAT files:

1. The header is parsed for image dimensions, DAC resolution, scan direction,
   and Z scaling.
2. The compressed payload is decompressed.
3. The forward topography channel (`FT`) is extracted.
4. The channel is scaled to native metre units using the Z calibration in the
   header.
5. Trailing blank rows are removed when present.

The flattened result for DAT input is written as a native-scale float32 TIFF.

## Normalization

The model-selection thresholds were tuned for data near a `[0, 1]` numerical
scale. TIFF and DAT data often contain very small native float values, so they
are normalized before fitting:

```text
normalized = (native - robust_offset) / robust_scale
```

The default robust offset and scale are the 1st percentile and the 1st-to-99th
percentile span. This normalization is not clipping. It is an affine transform
used only for fitting and model selection.

Native scaling is reapplied only when outputs are written:

- Flattened native output: `flattened_normalized * scale + offset`
- Removed background: `background_normalized * scale`

JPEGs keep their existing behavior: grayscale/luminance intensity divided by
255.

## Feature Masking

The algorithm excludes pixels that should not influence the background fit:

1. A smoothed image is used to compute gradient magnitude.
2. High-gradient pixels are marked as candidate step edges.
3. A median-filter residual is used to mark sharp local defects.
4. Edge and defect masks are dilated by configurable pixel padding.
5. Remaining connected smooth regions become fitting regions.

The diagnostic panel `Fit regions; red excluded` shows these masks. Red pixels
are excluded from the fit. Colored regions are smooth areas used for fitting.

## Terrace-Preserving Fit

The central safeguard is that each connected smooth region receives its own
constant offset during fitting. The fitted model is:

```text
height = polynomial_background(x, y) + region_offset(region)
```

Only the polynomial background is subtracted. Region offsets are not subtracted.
This means different terraces are not artificially forced to the same height.

## Candidate Models

Three candidate outputs are evaluated:

1. `none`: no correction.
2. `plane`: linear background in `x` and `y`.
3. `quadratic`: plane plus `x^2`, `xy`, and `y^2` curvature terms.

Plane and quadratic fits are robustly weighted so remaining outliers have
limited influence.

## Model Selection

For each candidate, the algorithm computes within-region residuals after
subtracting only each region's median. This measures residual slow background
without penalizing preserved terrace offsets.

Selection rules:

- Prefer no correction unless a correction improves the residual enough.
- Prefer plane over quadratic unless curvature gives an additional meaningful
  improvement.
- Reject tiny numerical corrections that do not remove a meaningful background
  range.

The JSON report records the selected correction, residual scores, fit-pixel
counts, mask thresholds, warnings, and output paths.

## Outputs

For each input, the script writes:

- Flattened preview PNG.
- Removed-background PNG.
- Fit-mask PNG.
- Diagnostic PNG.
- JSON report.
- Native scientific output:
  - `.tiff` for DAT/TIFF inputs.
  - `.npy` for JPEG intensity-proxy inputs.

The GUI's `Export accepted` command exports only image files:

- Flattened TIFF for DAT/TIFF inputs.
- Flattened JPEG for JPEG inputs.

## Limitations

- JPEG inputs are display products, not calibrated scientific height data.
- DAT parsing currently uses only forward topography.
- Sine scan mode is not supported.
- The algorithm is not a substitute for manual inspection; the GUI is intended
  to support review, subset reruns, and QC.
