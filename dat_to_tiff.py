#!/usr/bin/env python3
"""
Convert created .dat exports to TIFF stacks.

This follows the same parsing and scaling pattern as dats_to_pngs.py, but writes
float32 TIFFs instead of 8-bit PNG previews so the converted data retain more
numeric detail.
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np
import tifffile


CHANNELS: tuple[str, ...] = ("FT", "FC", "BT", "BC")


def _f(x, default=None):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _i(x, default=None):
    try:
        return int(float(str(x).replace(",", ".")))
    except Exception:
        return default


def parse_header(hb: bytes) -> dict[str, str]:
    hdr: dict[str, str] = {}
    for line in hb.splitlines():
        if b"=" in line:
            k, v = line.split(b"=", 1)
            key = k.decode("ascii", "ignore").split("/")[-1].strip()
            val = v.decode("ascii", "ignore").strip()
            hdr[key] = val
    return hdr


def find_hdr(hdr: dict[str, str], hint: str, default=None):
    for k in hdr:
        if hint.lower() in k.lower():
            return hdr[k]
    return default


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "image"


def get_dac_bits(hdr: dict[str, str], default: int = 20) -> int:
    raw = find_hdr(hdr, "DAC-Type", None)
    if raw is None:
        return default
    match = re.search(r"\d+", str(raw))
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass
    return default


def load_dat_stack(dat_path: Path) -> tuple[np.ndarray, tuple[str, ...], dict[str, str], int, int]:
    raw = dat_path.read_bytes()
    hb, comp = raw.split(b"DATA", 1)
    hdr = parse_header(hb)

    nx = _i(find_hdr(hdr, "Num.X", 0), 0)
    ny = _i(find_hdr(hdr, "Num.Y", 0), 0)
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid dimensions in header for {dat_path.name}")

    if _i(find_hdr(hdr, "ScanmodeSine", 0), 0) != 0:
        raise NotImplementedError("Sine scan mode not supported yet")

    payload = zlib.decompress(comp)
    total_floats = len(payload) // 4

    num_channels = None
    for candidate in (4, 2):
        if total_floats >= candidate * nx * ny:
            num_channels = candidate
            break
    if num_channels is None:
        raise ValueError(f"Payload too small for {dat_path.name}")

    arr = np.frombuffer(payload, dtype="<f4", count=num_channels * ny * nx).copy()
    stack = arr.reshape((num_channels, ny, nx)).astype(np.float32, copy=False)
    return stack, CHANNELS[:num_channels], hdr, nx, ny


def trim_stack(stack: np.ndarray) -> tuple[np.ndarray, int]:
    ch0 = stack[0]
    valid_mask = np.logical_and(~np.isnan(ch0), ch0 != 0)
    rows, cols = np.where(valid_mask.reshape(ch0.shape))
    if rows.size:
        last_row = int(rows.max())
        new_ny = last_row + 1 if cols[rows == last_row].max() == (ch0.shape[1] - 1) else last_row
    else:
        new_ny = ch0.shape[0]
    return stack[:, :new_ny, :], new_ny


def scale_stack(stack: np.ndarray, hdr: dict[str, str]) -> tuple[np.ndarray, dict[str, float]]:
    scaled = stack.astype(np.float32, copy=True)

    dac_bits = get_dac_bits(hdr, default=20)
    v_per_dac = 10.0 / (2**dac_bits)

    dz_ang_per_dac = _f(find_hdr(hdr, "Dacto[A]z", None), None)
    if dz_ang_per_dac is None:
        gain_z = _f(find_hdr(hdr, "GainZ", 10.0), 10.0)
        z_piezo = _f(find_hdr(hdr, "ZPiezoconst", 19.2), 19.2)
        dz_ang_per_dac = v_per_dac * gain_z * z_piezo * 1e2
    z_scale_m_per_dac = dz_ang_per_dac * 1e-10

    gain_pow = _f(find_hdr(hdr, "GainPre", _f(find_hdr(hdr, "GainPre 10^", 9), 9)), 9)
    preamp = 10.0 ** gain_pow
    i_scale_a_per_dac = (-1.0) * (v_per_dac / preamp)

    for idx in range(scaled.shape[0]):
        if idx % 2 == 0:
            scaled[idx] *= z_scale_m_per_dac
        else:
            scaled[idx] *= i_scale_a_per_dac

    if str(find_hdr(hdr, "ScanYDirec", "1")).strip() == "1":
        scaled = np.flip(scaled, axis=1)

    meta = {
        "dac_bits": float(dac_bits),
        "v_per_dac": float(v_per_dac),
        "z_scale_m_per_dac": float(z_scale_m_per_dac),
        "i_scale_a_per_dac": float(i_scale_a_per_dac),
    }
    return scaled, meta


def write_hdr_text(hdr_lines: bytes, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as out:
        for line in hdr_lines.splitlines():
            if b"=" in line:
                key, val = line.split(b"=", 1)
                field = key.decode("ascii", "ignore").split("/")[-1].strip()
                out.write(f"{field}: {val.decode('ascii', 'ignore').strip()}\n")


def convert_dat_to_tiff(dat_path: Path, out_root: Path) -> dict[str, object]:
    raw = dat_path.read_bytes()
    hb, _ = raw.split(b"DATA", 1)

    stack, channel_names, hdr, nx, ny = load_dat_stack(dat_path)
    trimmed_stack, kept_rows = trim_stack(stack)
    scaled_stack, scale_meta = scale_stack(trimmed_stack, hdr)

    origin_upper = str(find_hdr(hdr, "ScanYDirec", "1")).strip() == "1"

    out_dir = out_root / sanitize(dat_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_hdr_text(hb, out_dir / "hdr.txt")

    tiff_paths: list[str] = []
    for idx, (channel_name, arr_) in enumerate(zip(channel_names, scaled_stack, strict=True)):
        dr = "forward" if idx % 2 == 0 else "backward"
        disp = np.fliplr(arr_) if dr == "backward" else arr_
        if origin_upper:
            disp = np.flipud(disp)

        tiff_name = f"img_{idx:02d}_{sanitize(channel_name)}_{dr}.tiff"
        tiff_path = out_dir / tiff_name
        tifffile.imwrite(tiff_path, disp.astype(np.float32, copy=False), dtype=np.float32)
        tiff_paths.append(str(tiff_path))

    meta = {
        "input_path": str(dat_path),
        "output_dir": str(out_dir),
        "input_size": [nx, ny],
        "channels": list(channel_names),
        "kept_rows": int(kept_rows),
        "origin_upper": origin_upper,
        "scale": scale_meta,
        "tiff_paths": tiff_paths,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def iter_dat_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".dat" else []
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".dat")
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .dat files in ./in to float32 TIFFs in ./intiff.")
    parser.add_argument("--input_dir", default="in", help="Input .dat file or folder of .dat files.")
    parser.add_argument("--output_dir", default="intiff", help="Folder for TIFF outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    paths = iter_dat_paths(input_path)
    if not paths:
        raise SystemExit(f"No .dat files found in {input_path}")

    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            meta = convert_dat_to_tiff(path, output_root)
            print(f"[OK] {path.name} -> {meta['output_dir']}")
        except Exception as exc:
            failures.append({"input": str(path), "error": repr(exc)})
            print(f"[FAIL] {path.name}: {exc}")

    batch_report = {
        "input_path": str(input_path),
        "output_dir": str(output_root),
        "n_inputs": len(paths),
        "n_failures": len(failures),
        "failures": failures,
    }
    with (output_root / "batch_report.json").open("w", encoding="utf-8") as f:
        json.dump(batch_report, f, indent=2)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
