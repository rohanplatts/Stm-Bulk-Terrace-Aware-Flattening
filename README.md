# STM Autoflatten

Conservative batch flattening for STM topography images with a GUI review
workflow. The software is designed for atomic-resolution copper-style STM data
with terraces, step edges, drift, tilt, and scanner bow.

The key scientific constraint is that real terrace step heights should be
preserved. The algorithm masks step edges and defects, fits slow backgrounds
with independent per-region offsets, and subtracts only the slow polynomial
background.

## What It Does

- Reads `.dat`, `.tif/.tiff`, `.jpg/.jpeg`.
- For `.dat`, parses only forward topography (`FT`) and scales to native metre
  units.
- Processes a whole folder automatically.
- Chooses no correction, plane correction, or quadratic correction.
- Preserves terrace offsets instead of forcing terraces to equal height.
- Writes diagnostics and per-image JSON reports.
- Provides a GUI for review, parameter tuning, subset reruns, QC, and export.

## Recommended Workflow

For scientific work, use `.dat` or float32 TIFF input. JPEG input is supported
for convenience, but it is a display proxy rather than calibrated height data.

1. Put DAT files in `in/`.
2. Launch the GUI.
3. Run all images.
4. Review the scrollable image table.
5. Mark images as `Accepted`, `Needs Review`, or `Rejected`.
6. Tune masking parameters and rerun selected images if needed.
7. Export accepted outputs.

For DAT/TIFF inputs, accepted exports are flattened float32 TIFFs in native
height units.

## Installation

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional editable install:

```powershell
python -m pip install -e .
```

## Launch the GUI

On Windows, double-click:

```text
launch_stm_flatten_gui.bat
```

Or run:

```powershell
python .\stm_flatten_gui.py
```

If installed with `pip install -e .`:

```powershell
stm-flatten-gui
```

## Command-Line Flattening

```powershell
python .\stm_flatten.py --input_folder .\in --output_folder .\out
```

If `.\in` contains `.dat` files, those are processed preferentially. Converted
TIFF/JPEG files in the same folder are ignored by default to avoid duplicate
processing.

## Generate Demo Comparisons

The demo generator compares:

1. Forward-topography display.
2. Naive global plane flattening.
3. Naive global quadratic flattening.
4. `stm_flatten` masked flattening.

```powershell
python .\file_dedicated_to_obtaining_demo_files.py --input_dir .\in --output_dir .\demo
```

The generated examples are in `demo/`.

## Convert DAT to TIFF

This is optional because `stm_flatten.py` can now read DAT files directly.

```powershell
python .\dat_to_tiff.py --input_dir .\in --output_dir .\intiff
```

## Important Files

- `stm_flatten.py`: scientific flattening pipeline and CLI.
- `stm_flatten_gui.py`: operator GUI.
- `dat_to_tiff.py`: standalone DAT-to-TIFF converter.
- `file_dedicated_to_obtaining_demo_files.py`: demo comparison generator.
- `docs/ALGORITHM.md`: procedural explanation of the flattening method.
- `demo/`: generated comparison images.

## Outputs

Per processed image:

- `*_flattened.tiff`: native-scale scientific output for DAT/TIFF inputs.
- `*_flattened.npy`: scientific-ish output for JPEG intensity proxies.
- `*_flattened.png`: quick-look preview.
- `*_background.png`: removed background preview.
- `*_fit_mask.png`: fitting mask preview.
- `*_diagnostic.png`: multi-panel diagnostic.
- `*_report.json`: parameter and decision report.

The GUI `Export accepted` command exports only user-facing image outputs:

- TIFF for DAT/TIFF sources.
- JPEG for JPEG sources.

## Notes For Contributors

Raw DAT/TIFF inputs and generated run folders are ignored by default. Demo PNGs
are intentionally trackable so the repository can demonstrate why the method is
useful.

Run quick checks:

```powershell
python -m py_compile .\stm_flatten.py .\stm_flatten_gui.py .\dat_to_tiff.py .\file_dedicated_to_obtaining_demo_files.py
```

## License

MIT. See `LICENSE`.
