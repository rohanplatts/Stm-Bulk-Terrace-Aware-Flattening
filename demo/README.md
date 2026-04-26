# Demo outputs

This folder contains generated comparison PNGs for the example DAT files used
during development.

Each `*_demo.png` has four panels:

1. Forward-topography display.
2. Naive global plane flattening.
3. Naive global quadratic flattening.
4. `stm_flatten` masked flattening.

Regenerate the images with:

```powershell
python .\file_dedicated_to_obtaining_demo_files.py --input_dir .\in --output_dir .\demo
```

Raw `.dat` files are intentionally not tracked by default. Place local DAT files
in `in/` before regenerating.
