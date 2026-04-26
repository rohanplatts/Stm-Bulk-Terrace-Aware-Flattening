#!/usr/bin/env python3
"""
Interactive GUI for folder-based STM flattening.

The scientific pipeline remains in stm_flatten.py. This GUI is responsible for
operator workflow: batch/subset processing, parameter presets, QC state, export,
and a live scrollable image-review table.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageTk

import stm_flatten


APP_TITLE = "STM Folder Flattening Review"
SUPPORTED_SUFFIXES = {".dat", ".jpg", ".jpeg", ".tif", ".tiff"}
QC_STATES = ("Needs Review", "Accepted", "Rejected")
CORRECTION_FILTERS = ("All", "none", "plane", "quadratic", "failed", "pending")
QC_FILTERS = ("All",) + QC_STATES

OUTPUT_SUFFIXES = (
    "_flattened.png",
    "_flattened.npy",
    "_flattened.tiff",
    "_background.png",
    "_fit_mask.png",
    "_diagnostic.png",
    "_report.json",
)

PARAMETERS: dict[str, dict[str, Any]] = {
    "edge_percentile": {
        "default": 90.0,
        "type": float,
        "label": "Step-edge sensitivity percentile",
        "group": "Masking",
        "help": "Lower values exclude more high-gradient pixels from the background fit.",
    },
    "edge_mad_multiplier": {
        "default": 5.0,
        "type": float,
        "label": "Step-edge MAD multiplier",
        "group": "Masking",
        "help": "Lower values make edge masking more aggressive.",
    },
    "edge_dilate_px": {
        "default": 7,
        "type": int,
        "label": "Step-edge padding (px)",
        "group": "Masking",
        "help": "Extra pixels excluded around detected step edges.",
    },
    "gradient_sigma": {
        "default": 1.0,
        "type": float,
        "label": "Gradient smoothing sigma",
        "group": "Masking",
        "help": "Smoothing before edge detection. Larger values target broader features.",
    },
    "defect_percentile": {
        "default": 98.5,
        "type": float,
        "label": "Defect sensitivity percentile",
        "group": "Masking",
        "help": "Lower values exclude more point-like defects from fitting.",
    },
    "defect_mad_multiplier": {
        "default": 8.0,
        "type": float,
        "label": "Defect MAD multiplier",
        "group": "Masking",
        "help": "Lower values make defect masking more aggressive.",
    },
    "defect_filter_size": {
        "default": 5,
        "type": int,
        "label": "Defect filter size",
        "group": "Advanced Masking",
        "help": "Median-filter window used to identify local spikes and pits.",
    },
    "defect_dilate_px": {
        "default": 1,
        "type": int,
        "label": "Defect padding (px)",
        "group": "Masking",
        "help": "Extra pixels excluded around detected local defects.",
    },
    "min_component_pixels": {
        "default": 160,
        "type": int,
        "label": "Min terrace region pixels",
        "group": "Advanced Masking",
        "help": "Small smooth regions below this size are ignored in the fit.",
    },
    "min_component_fraction": {
        "default": 0.0025,
        "type": float,
        "label": "Min terrace region fraction",
        "group": "Advanced Masking",
        "help": "Fractional lower bound for retained smooth fit regions.",
    },
    "max_components": {
        "default": 80,
        "type": int,
        "label": "Max terrace regions",
        "group": "Advanced Masking",
        "help": "Upper limit on separate smooth regions used in the fit.",
    },
    "min_fit_fraction": {
        "default": 0.08,
        "type": float,
        "label": "Minimum fit fraction",
        "group": "Advanced Masking",
        "help": "Fallback threshold if masking leaves too few fit pixels.",
    },
    "max_fit_pixels": {
        "default": 120000,
        "type": int,
        "label": "Max fit pixels",
        "group": "Advanced Masking",
        "help": "Subsampling cap for large images.",
    },
    "plane_min_relative_improvement": {
        "default": 0.05,
        "type": float,
        "label": "Plane improvement threshold",
        "group": "Model Selection",
        "help": "Minimum relative improvement needed before accepting a plane.",
    },
    "quadratic_min_relative_improvement": {
        "default": 0.08,
        "type": float,
        "label": "Quadratic vs plane threshold",
        "group": "Model Selection",
        "help": "Minimum extra relative improvement needed before accepting quadratic over plane.",
    },
    "quadratic_from_none_min_relative_improvement": {
        "default": 0.12,
        "type": float,
        "label": "Quadratic vs none threshold",
        "group": "Model Selection",
        "help": "Minimum relative improvement for quadratic if plane was not accepted.",
    },
    "min_abs_improvement": {
        "default": 0.002,
        "type": float,
        "label": "Minimum absolute improvement",
        "group": "Model Selection",
        "help": "Absolute residual improvement floor; normalized units.",
    },
    "min_background_range": {
        "default": 0.005,
        "type": float,
        "label": "Minimum background range",
        "group": "Model Selection",
        "help": "Rejects tiny numerical corrections that are unlikely to matter.",
    },
    "random_seed": {
        "default": 12345,
        "type": int,
        "label": "Random seed",
        "group": "Model Selection",
        "help": "Used only when large images require fit-pixel subsampling.",
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "Aggressive masking": {
        "edge_percentile": 85.0,
        "edge_mad_multiplier": 3.5,
        "edge_dilate_px": 10,
        "defect_percentile": 97.5,
        "defect_mad_multiplier": 6.0,
        "defect_dilate_px": 2,
    },
    "Permissive masking": {
        "edge_percentile": 95.0,
        "edge_mad_multiplier": 7.0,
        "edge_dilate_px": 4,
        "defect_percentile": 99.3,
        "defect_mad_multiplier": 10.0,
        "defect_dilate_px": 0,
    },
}

COLUMNS = (
    ("original", "Original"),
    ("flattened", "Flattened"),
    ("fit_regions", "Fit regions; red excluded"),
    ("background", "Removed background"),
    ("residual", "Within-region residual"),
)


@dataclass
class ImageJob:
    path: Path
    status: str = "pending"
    qc_state: str = "Needs Review"
    correction: str = "pending"
    reason: str = ""
    report_path: Path | None = None
    diagnostic_path: Path | None = None
    scientific_output_path: Path | None = None
    report: dict[str, Any] | None = None
    error: str = ""


class ScrollableFrame(ttk.Frame):
    """A ttk frame with a vertical scrollbar for long parameter forms."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.inner = ttk.Frame(self.canvas, padding=8)
        self.inner.columnconfigure(1, weight=1)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind("<Configure>", self._fit_inner_width)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    def _update_scrollregion(self, _event: tk.Event | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fit_inner_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")
        return "break"


class ReviewTable(ttk.Frame):
    """Virtual scrolling canvas table for image rows.

    Only visible rows are drawn. This avoids creating thousands of Tk image
    objects for large folders and makes zooming responsive enough for review.
    """

    def __init__(self, master: tk.Misc, app: "FlattenGui") -> None:
        super().__init__(master)
        self.app = app
        self.tile_size = 190
        self.meta_width = 190
        self.gap = 12
        self.header_height = 42
        self.row_pad_y = 12
        self.visible_photos: list[ImageTk.PhotoImage] = []
        self._redraw_after_id: str | None = None
        self.vertical_wheel_pixels = 72
        self.horizontal_wheel_pixels = 96

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.header = tk.Canvas(self, height=self.header_height, bg="#20252b", highlightthickness=0)
        self.header.grid(row=0, column=0, sticky="ew")

        self.canvas = tk.Canvas(self, bg="#f5f7f9", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._yview)
        self.vbar.grid(row=1, column=1, sticky="ns")
        self.hbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._xview)
        self.hbar.grid(row=2, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.header.configure(xscrollcommand=self.hbar.set)

        self.canvas.bind("<Configure>", lambda _event: self.schedule_redraw())
        self.header.bind("<Configure>", lambda _event: self.draw_header())
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Double-1>", self._on_double_click)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.header.bind("<MouseWheel>", self._on_mousewheel)
        self.draw_header()

    @property
    def row_height(self) -> int:
        return self.tile_size + 2 * self.row_pad_y

    @property
    def column_width(self) -> int:
        return self.tile_size + self.gap

    @property
    def content_width(self) -> int:
        return self.meta_width + self.gap + len(COLUMNS) * self.column_width

    def set_zoom(self, tile_size: int) -> None:
        self.tile_size = max(110, min(420, int(tile_size)))
        self.app.clear_image_cache()
        self.update_scrollregion()
        self.draw_header()
        self.redraw()

    def zoom_by(self, delta: int) -> None:
        self.set_zoom(self.tile_size + delta)

    def update_scrollregion(self) -> None:
        height = max(len(self.app.filtered_indices) * self.row_height, 1)
        width = max(self.content_width, 1)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.header.configure(scrollregion=(0, 0, width, self.header_height))

    def draw_header(self) -> None:
        self.header.delete("all")
        self.header.create_rectangle(0, 0, self.content_width, self.header_height, fill="#20252b", outline="")
        self.header.create_text(12, 22, text="File / QC / model", anchor="w", fill="white", font=("Segoe UI", 10, "bold"))
        for col_idx, (_key, label) in enumerate(COLUMNS):
            x0 = self.meta_width + self.gap + col_idx * self.column_width
            self.header.create_text(
                x0 + self.tile_size // 2,
                22,
                text=label,
                anchor="center",
                fill="white",
                font=("Segoe UI", 10, "bold"),
            )

    def redraw(self) -> None:
        self.update_scrollregion()
        self.canvas.delete("all")
        self.visible_photos.clear()

        if not self.app.filtered_indices:
            self.canvas.create_text(
                40,
                40,
                text="No images match the current filters.",
                anchor="nw",
                fill="#333333",
                font=("Segoe UI", 12),
            )
            return

        top = int(self.canvas.canvasy(0))
        bottom = int(self.canvas.canvasy(max(self.canvas.winfo_height(), 1)))
        first = max(top // self.row_height - 1, 0)
        last = min(bottom // self.row_height + 2, len(self.app.filtered_indices))

        for display_row in range(first, last):
            job_index = self.app.filtered_indices[display_row]
            job = self.app.jobs[job_index]
            self._draw_row(display_row, job_index, job)

    def _draw_row(self, display_row: int, job_index: int, job: ImageJob) -> None:
        y0 = display_row * self.row_height
        y1 = y0 + self.row_height
        selected = job_index in self.app.selected_indices
        fill = "#dbeafe" if selected else ("#ffffff" if display_row % 2 == 0 else "#f0f3f6")
        self.canvas.create_rectangle(0, y0, self.content_width, y1, fill=fill, outline="#d0d7de")

        qc_color = {
            "Accepted": "#147a35",
            "Needs Review": "#9a6700",
            "Rejected": "#b42318",
        }.get(job.qc_state, "#444444")
        text = self.app.row_text(job)
        self.canvas.create_text(
            12,
            y0 + 14,
            text=text,
            anchor="nw",
            width=self.meta_width - 24,
            fill="#111111",
            font=("Segoe UI", 9),
        )
        self.canvas.create_rectangle(12, y1 - 22, 118, y1 - 6, fill=qc_color, outline="")
        self.canvas.create_text(65, y1 - 14, text=job.qc_state, fill="white", font=("Segoe UI", 8, "bold"))

        for col_idx, (key, _label) in enumerate(COLUMNS):
            x0 = self.meta_width + self.gap + col_idx * self.column_width
            tile = self.app.get_tile(job, key, self.tile_size)
            if tile is not None:
                self.visible_photos.append(tile)
                self.canvas.create_image(x0, y0 + self.row_pad_y, image=tile, anchor="nw")
            else:
                self._draw_placeholder(x0, y0 + self.row_pad_y, key)
            self.canvas.create_rectangle(
                x0,
                y0 + self.row_pad_y,
                x0 + self.tile_size,
                y0 + self.row_pad_y + self.tile_size,
                outline="#9aa4af",
            )

    def _draw_placeholder(self, x: int, y: int, label: str) -> None:
        self.canvas.create_rectangle(x, y, x + self.tile_size, y + self.tile_size, fill="#e5e7eb", outline="#9aa4af")
        self.canvas.create_text(
            x + self.tile_size // 2,
            y + self.tile_size // 2,
            text=label.replace("_", " "),
            fill="#555555",
            font=("Segoe UI", 9),
        )

    def _display_row_from_event(self, event: tk.Event) -> int | None:
        y = int(self.canvas.canvasy(event.y))
        display_row = y // self.row_height
        if display_row < 0 or display_row >= len(self.app.filtered_indices):
            return None
        return display_row

    def _on_click(self, event: tk.Event) -> None:
        display_row = self._display_row_from_event(event)
        if display_row is None:
            return
        job_index = self.app.filtered_indices[display_row]
        ctrl = bool(event.state & 0x0004)
        shift = bool(event.state & 0x0001)
        self.app.select_index(job_index, ctrl=ctrl, shift=shift)

    def _on_double_click(self, event: tk.Event) -> None:
        display_row = self._display_row_from_event(event)
        if display_row is None:
            return
        job_index = self.app.filtered_indices[display_row]
        self.app.open_job_diagnostic(self.app.jobs[job_index])

    def _on_mousewheel(self, event: tk.Event) -> str:
        ctrl = bool(event.state & 0x0004)
        shift = bool(event.state & 0x0001)
        if ctrl:
            self.zoom_by(24 if event.delta > 0 else -24)
            return "break"
        if shift:
            self.scroll_x_pixels(-self._wheel_steps(event.delta) * self.horizontal_wheel_pixels)
            return "break"
        self.scroll_y_pixels(-self._wheel_steps(event.delta) * self.vertical_wheel_pixels)
        return "break"

    def _yview(self, *args: str) -> None:
        self.canvas.yview(*args)
        self.schedule_redraw()

    def _xview(self, *args: str) -> None:
        self.canvas.xview(*args)
        self.header.xview(*args)
        self.schedule_redraw()

    def schedule_redraw(self, delay_ms: int = 12) -> None:
        if self._redraw_after_id is not None:
            return
        self._redraw_after_id = self.after(delay_ms, self._run_scheduled_redraw)

    def _run_scheduled_redraw(self) -> None:
        self._redraw_after_id = None
        self.redraw()

    def _wheel_steps(self, delta: int) -> float:
        if delta == 0:
            return 0.0
        # Windows mouse wheels normally send +/-120. Some touchpads send
        # smaller deltas; scale those more gently so they remain usable.
        if abs(delta) >= 120:
            return delta / 120.0
        if abs(delta) >= 10:
            return delta / 30.0
        return float(delta)

    def scroll_y_pixels(self, pixels: float) -> None:
        self.update_scrollregion()
        scroll_height = max(len(self.app.filtered_indices) * self.row_height, 1)
        viewport = max(self.canvas.winfo_height(), 1)
        max_top = max(scroll_height - viewport, 0)
        current_top = float(self.canvas.canvasy(0))
        new_top = min(max(current_top + pixels, 0.0), float(max_top))
        self.canvas.yview_moveto(new_top / scroll_height if scroll_height else 0.0)
        self.schedule_redraw()

    def scroll_x_pixels(self, pixels: float) -> None:
        self.update_scrollregion()
        scroll_width = max(self.content_width, 1)
        viewport = max(self.canvas.winfo_width(), 1)
        max_left = max(scroll_width - viewport, 0)
        current_left = float(self.canvas.canvasx(0))
        new_left = min(max(current_left + pixels, 0.0), float(max_left))
        fraction = new_left / scroll_width if scroll_width else 0.0
        self.canvas.xview_moveto(fraction)
        self.header.xview_moveto(fraction)
        self.schedule_redraw()


class FlattenGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1540x920")
        self.minsize(1150, 740)

        self.input_var = tk.StringVar(value=str((Path.cwd() / "in").resolve()))
        self.output_var = tk.StringVar(value=str((Path.cwd() / "out_gui").resolve()))
        self.status_var = tk.StringVar(value="Scan a folder, run flattening, then review images directly in the table.")
        self.reset_outputs_var = tk.BooleanVar(value=True)
        self.qc_filter_var = tk.StringVar(value="All")
        self.correction_filter_var = tk.StringVar(value="All")
        self.zoom_var = tk.StringVar(value="190")
        self.preset_var = tk.StringVar(value="Aggressive masking")
        self.param_vars = {name: tk.StringVar(value=str(spec["default"])) for name, spec in PARAMETERS.items()}

        self.jobs: list[ImageJob] = []
        self.filtered_indices: list[int] = []
        self.selected_indices: set[int] = set()
        self.anchor_index: int | None = None
        self.image_cache: dict[tuple[str, str, int, float], ImageTk.PhotoImage] = {}
        self.worker_thread: threading.Thread | None = None
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self._build_ui()
        self.after(150, self._poll_worker_queue)
        self.scan_input()

    def _build_ui(self) -> None:
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        self._build_folder_bar()
        self._build_action_bar()

        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)

        review_frame = ttk.Frame(pane)
        review_frame.rowconfigure(1, weight=1)
        review_frame.columnconfigure(0, weight=1)
        pane.add(review_frame, weight=5)

        self._build_filter_bar(review_frame)
        self.table = ReviewTable(review_frame, self)
        self.table.grid(row=1, column=0, sticky="nsew")

        side = ttk.Frame(pane)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(0, weight=1)
        pane.add(side, weight=1)
        self._build_parameter_panel(side)

        status_bar = ttk.Frame(self, padding=(8, 4, 8, 8))
        status_bar.grid(row=3, column=0, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(status_bar, mode="determinate", length=260)
        self.progress.grid(row=0, column=1, sticky="e")

    def _build_folder_bar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 8, 8, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)

        ttk.Label(bar, text="Input").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(bar, textvariable=self.input_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(bar, text="Browse", command=self.choose_input).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(bar, text="Scan", command=self.scan_input).grid(row=0, column=3, padx=(6, 0))

        ttk.Label(bar, text="Output").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
        ttk.Entry(bar, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(bar, text="Browse", command=self.choose_output).grid(row=1, column=2, padx=(6, 0), pady=(6, 0))
        ttk.Button(bar, text="Open", command=self.open_output_folder).grid(row=1, column=3, padx=(6, 0), pady=(6, 0))

    def _build_action_bar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 4, 8, 4))
        bar.grid(row=1, column=0, sticky="ew")
        for idx in range(12):
            bar.columnconfigure(idx, weight=0)

        ttk.Button(bar, text="Run all", command=self.run_all).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(bar, text="Run selected", command=self.run_selected).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(bar, text="Rerun selected", command=self.run_selected).grid(row=0, column=2, padx=(0, 12))
        ttk.Checkbutton(bar, text="Reset outputs before rerun", variable=self.reset_outputs_var).grid(
            row=0, column=3, padx=(0, 12)
        )
        ttk.Button(bar, text="Accept", command=lambda: self.set_qc_for_selection("Accepted")).grid(row=0, column=4)
        ttk.Button(bar, text="Needs Review", command=lambda: self.set_qc_for_selection("Needs Review")).grid(
            row=0, column=5, padx=(6, 0)
        )
        ttk.Button(bar, text="Reject", command=lambda: self.set_qc_for_selection("Rejected")).grid(
            row=0, column=6, padx=(6, 12)
        )
        ttk.Button(bar, text="Export accepted", command=self.export_accepted).grid(row=0, column=7, padx=(0, 12))
        ttk.Button(bar, text="Open diagnostic", command=self.open_selected_diagnostic).grid(row=0, column=8)

        ttk.Label(bar, text="Zoom").grid(row=0, column=9, padx=(14, 4))
        ttk.Button(bar, text="-", width=3, command=lambda: self.table.zoom_by(-24)).grid(row=0, column=10)
        ttk.Button(bar, text="+", width=3, command=lambda: self.table.zoom_by(24)).grid(row=0, column=11, padx=(4, 0))

    def _build_filter_bar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(bar, text="QC").grid(row=0, column=0, padx=(0, 4))
        qc = ttk.Combobox(bar, textvariable=self.qc_filter_var, values=QC_FILTERS, state="readonly", width=14)
        qc.grid(row=0, column=1, padx=(0, 12))
        qc.bind("<<ComboboxSelected>>", lambda _event: self.apply_filters())

        ttk.Label(bar, text="Correction").grid(row=0, column=2, padx=(0, 4))
        corr = ttk.Combobox(
            bar,
            textvariable=self.correction_filter_var,
            values=CORRECTION_FILTERS,
            state="readonly",
            width=12,
        )
        corr.grid(row=0, column=3, padx=(0, 12))
        corr.bind("<<ComboboxSelected>>", lambda _event: self.apply_filters())

        ttk.Button(bar, text="Select visible", command=self.select_visible).grid(row=0, column=4)
        ttk.Button(bar, text="Select all", command=self.select_all).grid(row=0, column=5, padx=(6, 0))
        ttk.Button(bar, text="Clear selection", command=self.clear_selection).grid(row=0, column=6, padx=(6, 0))
        ttk.Label(
            bar,
            text="Tip: mouse wheel scrolls rows, Shift+wheel scrolls sideways, Ctrl+wheel zooms.",
            foreground="#555555",
        ).grid(row=0, column=7, padx=(18, 0), sticky="w")

    def _build_qc_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Selected Image", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        self.selected_text = tk.Text(frame, height=9, wrap="word")
        self.selected_text.grid(row=0, column=0, sticky="ew")
        self.selected_text.configure(state="disabled")

    def _build_parameter_panel(self, parent: ttk.Frame) -> None:
        outer = ttk.LabelFrame(parent, text="Parameters", padding=8)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(2, weight=1)
        outer.columnconfigure(0, weight=1)

        preset = ttk.Frame(outer)
        preset.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        preset.columnconfigure(0, weight=1)
        ttk.Combobox(preset, textvariable=self.preset_var, values=list(PRESETS), state="readonly").grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(preset, text="Apply", command=self.apply_preset).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(preset, text="Defaults", command=self.reset_defaults).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(preset, text="Save", command=self.save_settings).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Button(preset, text="Load", command=self.load_settings).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        notebook = ttk.Notebook(outer)
        notebook.grid(row=2, column=0, sticky="nsew")
        frames: dict[str, ttk.Frame] = {}
        for group in ("Masking", "Advanced Masking", "Model Selection"):
            scroller = ScrollableFrame(notebook)
            notebook.add(scroller, text=group)
            frames[group] = scroller.inner

        rows = {group: 0 for group in frames}
        for name, spec in PARAMETERS.items():
            frame = frames[spec["group"]]
            row = rows[spec["group"]]
            ttk.Label(frame, text=spec["label"]).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=self.param_vars[name], width=12).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=2)
            ttk.Label(frame, text=spec["help"], foreground="#666666", wraplength=280).grid(
                row=row + 1,
                column=0,
                columnspan=2,
                sticky="ew",
                pady=(0, 7),
            )
            rows[spec["group"]] += 2

    def choose_input(self) -> None:
        selected = filedialog.askdirectory(title="Choose input folder", initialdir=self.input_var.get() or os.getcwd())
        if selected:
            self.input_var.set(selected)
            self.scan_input()

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Choose output folder", initialdir=self.output_var.get() or os.getcwd())
        if selected:
            self.output_var.set(selected)
            self.scan_input()

    def open_output_folder(self) -> None:
        output = Path(self.output_var.get()).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        os.startfile(output)

    def scan_input(self) -> None:
        input_path = Path(self.input_var.get()).expanduser()
        paths = stm_flatten.input_paths(input_path)
        qc_state = self.load_qc_state()
        self.jobs = []
        for path in paths:
            job = ImageJob(path=path.resolve())
            job.qc_state = qc_state.get(str(job.path), "Needs Review")
            self.load_existing_report(job)
            self.jobs.append(job)
        self.selected_indices.clear()
        self.anchor_index = None
        self.clear_image_cache()
        self.apply_filters()
        self.status_var.set(f"Loaded {len(self.jobs)} image(s).")

    def apply_filters(self) -> None:
        qc_filter = self.qc_filter_var.get()
        correction_filter = self.correction_filter_var.get()
        filtered: list[int] = []
        for idx, job in enumerate(self.jobs):
            if qc_filter != "All" and job.qc_state != qc_filter:
                continue
            corr = job.correction if job.status != "failed" else "failed"
            if correction_filter != "All" and corr != correction_filter:
                continue
            filtered.append(idx)
        self.filtered_indices = filtered
        self.selected_indices = {idx for idx in self.selected_indices if idx in self.filtered_indices}
        self.table.redraw()
        self.update_selected_text()

    def load_existing_report(self, job: ImageJob) -> None:
        report_path = self.expected_report_path(job.path)
        if not report_path.exists():
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.apply_report_to_job(job, report)

    def expected_report_path(self, input_path: Path) -> Path:
        output_folder = Path(self.output_var.get()).expanduser().resolve()
        prefix = output_folder / stm_flatten.safe_stem(input_path)
        return prefix.with_name(prefix.name + "_report.json")

    def current_args(self) -> argparse.Namespace | None:
        values: dict[str, Any] = {}
        errors: list[str] = []
        for name, spec in PARAMETERS.items():
            raw = self.param_vars[name].get().strip()
            try:
                values[name] = spec["type"](raw)
            except ValueError:
                errors.append(f"{spec['label']}: {raw!r}")
        if errors:
            messagebox.showerror("Invalid parameters", "Fix these fields:\n\n" + "\n".join(errors))
            return None
        return argparse.Namespace(**values)

    def apply_preset(self) -> None:
        self.reset_defaults()
        for key, value in PRESETS.get(self.preset_var.get(), {}).items():
            self.param_vars[key].set(str(value))

    def reset_defaults(self) -> None:
        for name, spec in PARAMETERS.items():
            self.param_vars[name].set(str(spec["default"]))

    def save_settings(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save flattening settings",
            defaultextension=".json",
            filetypes=[("JSON settings", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = {name: var.get() for name, var in self.param_vars.items()}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_settings(self) -> None:
        path = filedialog.askopenfilename(
            title="Load flattening settings",
            filetypes=[("JSON settings", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, value in data.items():
            if name in self.param_vars:
                self.param_vars[name].set(str(value))

    def run_all(self) -> None:
        self.start_run(list(range(len(self.jobs))))

    def run_selected(self) -> None:
        if not self.selected_indices:
            messagebox.showinfo("No selection", "Select one or more rows first.")
            return
        self.start_run(sorted(self.selected_indices))

    def start_run(self, indices: list[int]) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Run in progress", "Wait for the current run to finish.")
            return
        if not indices:
            messagebox.showinfo("No images", "No images to process.")
            return
        args = self.current_args()
        if args is None:
            return
        output_folder = Path(self.output_var.get()).expanduser().resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        reset_outputs = self.reset_outputs_var.get()

        for idx in indices:
            self.jobs[idx].status = "queued"
            self.jobs[idx].error = ""
        self.table.redraw()
        self.progress.configure(value=0, maximum=len(indices))
        self.status_var.set(f"Running {len(indices)} image(s)...")

        self.worker_thread = threading.Thread(
            target=self.run_worker,
            args=(indices, output_folder, args, reset_outputs),
            daemon=True,
        )
        self.worker_thread.start()

    def run_worker(self, indices: list[int], output_folder: Path, args: argparse.Namespace, reset_outputs: bool) -> None:
        run_summary = {
            "output_folder": str(output_folder),
            "n_images": len(indices),
            "parameters": vars(args),
            "reset_outputs_before_run": reset_outputs,
            "images": [],
            "failures": [],
        }
        completed = 0
        for idx in indices:
            job = self.jobs[idx]
            try:
                self.result_queue.put(("status", (idx, "running")))
                if reset_outputs:
                    self.delete_outputs_for(job.path, output_folder)
                report = stm_flatten.process_image(job.path, output_folder, args)
                run_summary["images"].append(
                    {
                        "input": str(job.path),
                        "selected_correction": report.get("selected_correction", ""),
                        "selection_reason": report.get("selection_reason", ""),
                        "report_json": report.get("outputs", {}).get("report_json", ""),
                        "scientific_output": report.get("scientific_data_output", {}).get("path", ""),
                    }
                )
                completed += 1
                self.result_queue.put(("result", (idx, report, completed, len(indices))))
            except Exception as exc:
                completed += 1
                run_summary["failures"].append({"input": str(job.path), "error": repr(exc)})
                self.result_queue.put(("error", (idx, repr(exc), completed, len(indices))))

        run_summary["n_success"] = len(run_summary["images"])
        run_summary["n_failures"] = len(run_summary["failures"])
        try:
            (output_folder / "gui_last_run_report.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        except OSError as exc:
            self.result_queue.put(("message", f"Could not write GUI run report: {exc}"))
        self.result_queue.put(("done", None))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "status":
                    idx, status = payload
                    self.jobs[idx].status = status
                    self.status_var.set(f"{status}: {self.jobs[idx].path.name}")
                elif kind == "result":
                    idx, report, completed, total = payload
                    self.apply_report_to_job(self.jobs[idx], report)
                    self.progress.configure(value=completed, maximum=total)
                    self.status_var.set(f"Completed {completed}/{total}: {self.jobs[idx].path.name}")
                elif kind == "error":
                    idx, error, completed, total = payload
                    job = self.jobs[idx]
                    job.status = "failed"
                    job.correction = "failed"
                    job.error = error
                    self.progress.configure(value=completed, maximum=total)
                    self.status_var.set(f"Failed {job.path.name}: {error}")
                elif kind == "message":
                    self.status_var.set(str(payload))
                elif kind == "done":
                    self.save_qc_state()
                    self.apply_filters()
                    self.status_var.set("Run finished.")
                self.clear_image_cache()
                self.table.redraw()
                self.update_selected_text()
        except queue.Empty:
            pass
        self.after(150, self._poll_worker_queue)

    def delete_outputs_for(self, input_path: Path, output_folder: Path) -> None:
        prefix = output_folder / stm_flatten.safe_stem(input_path)
        resolved_output = output_folder.resolve()
        for suffix in OUTPUT_SUFFIXES:
            target = prefix.with_name(prefix.name + suffix)
            try:
                resolved_target = target.resolve()
                if resolved_target.parent == resolved_output and target.exists():
                    target.unlink()
            except OSError:
                pass

    def apply_report_to_job(self, job: ImageJob, report: dict[str, Any]) -> None:
        job.status = "done"
        job.report = report
        job.correction = str(report.get("selected_correction", ""))
        job.reason = str(report.get("selection_reason", ""))
        job.error = ""
        outputs = report.get("outputs", {})
        report_value = outputs.get("report_json")
        diagnostic_value = outputs.get("diagnostic_png")
        job.report_path = Path(report_value) if report_value else self.expected_report_path(job.path)
        job.diagnostic_path = Path(diagnostic_value) if diagnostic_value else None
        scientific_value = report.get("scientific_data_output", {}).get("path")
        job.scientific_output_path = Path(scientific_value) if scientific_value else None

    def select_index(self, job_index: int, ctrl: bool = False, shift: bool = False) -> None:
        if shift and self.anchor_index is not None:
            visible = self.filtered_indices
            if job_index in visible and self.anchor_index in visible:
                a = visible.index(self.anchor_index)
                b = visible.index(job_index)
                lo, hi = sorted((a, b))
                self.selected_indices = set(visible[lo : hi + 1])
        elif ctrl:
            if job_index in self.selected_indices:
                self.selected_indices.remove(job_index)
            else:
                self.selected_indices.add(job_index)
            self.anchor_index = job_index
        else:
            self.selected_indices = {job_index}
            self.anchor_index = job_index
        self.table.redraw()
        self.update_selected_text()

    def select_visible(self) -> None:
        self.selected_indices = set(self.filtered_indices)
        self.table.redraw()
        self.update_selected_text()

    def select_all(self) -> None:
        self.selected_indices = set(range(len(self.jobs)))
        self.apply_filters()
        self.status_var.set(f"Selected all {len(self.selected_indices)} loaded image(s).")

    def clear_selection(self) -> None:
        self.selected_indices.clear()
        self.anchor_index = None
        self.table.redraw()
        self.update_selected_text()

    def set_qc_for_selection(self, state: str) -> None:
        if not self.selected_indices:
            messagebox.showinfo("No selection", "Select one or more rows first.")
            return
        for idx in self.selected_indices:
            self.jobs[idx].qc_state = state
        self.save_qc_state()
        self.apply_filters()

    def update_selected_text(self) -> None:
        if not hasattr(self, "selected_text"):
            return
        text = ""
        if self.selected_indices:
            idx = sorted(self.selected_indices)[0]
            text = self.row_text(self.jobs[idx], verbose=True)
            if len(self.selected_indices) > 1:
                text += f"\n\n{len(self.selected_indices)} rows selected."
        self.selected_text.configure(state="normal")
        self.selected_text.delete("1.0", "end")
        self.selected_text.insert("1.0", text)
        self.selected_text.configure(state="disabled")

    def row_text(self, job: ImageJob, verbose: bool = False) -> str:
        lines = [
            job.path.name,
            f"{job.qc_state}",
            f"{job.status} / {job.correction}",
        ]
        if job.report:
            candidates = job.report.get("candidates", [])
            if candidates:
                score_bits = []
                for c in candidates:
                    try:
                        score_bits.append(f"{c.get('name')} {float(c.get('score_mad_sigma')):.2g}")
                    except (TypeError, ValueError):
                        pass
                if score_bits:
                    lines.append(", ".join(score_bits))
            warnings = job.report.get("masking", {}).get("warnings", [])
            if warnings:
                lines.append("Warnings: " + "; ".join(warnings[:1]))
            sci = job.report.get("scientific_data_output", {})
            if verbose and sci.get("path"):
                lines.append(f"Scientific output: {sci.get('path')}")
        if job.reason and (verbose or len(job.reason) < 120):
            lines.append(f"Reason: {job.reason}")
        if job.error:
            lines.append(f"Error: {job.error}")
        return "\n".join(lines)

    def qc_state_path(self) -> Path:
        return Path(self.output_var.get()).expanduser().resolve() / "gui_qc_state.json"

    def load_qc_state(self) -> dict[str, str]:
        path = self.qc_state_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in raw.items() if str(v) in QC_STATES}

    def save_qc_state(self) -> None:
        output = Path(self.output_var.get()).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        data = {str(job.path): job.qc_state for job in self.jobs}
        self.qc_state_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def export_accepted(self) -> None:
        accepted = [job for job in self.jobs if job.qc_state == "Accepted"]
        if not accepted:
            messagebox.showinfo("No accepted images", "Mark images as Accepted first.")
            return
        target_text = filedialog.askdirectory(title="Choose export folder")
        if not target_text:
            return
        target = Path(target_text).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)

        exported, missing = self.export_jobs_as_images(accepted, target)
        message = f"Exported {exported} accepted image file(s) to {target}."
        if missing:
            message += f"\nCould not export {len(missing)} accepted image(s):\n" + "\n".join(missing[:10])
        messagebox.showinfo("Export complete", message)

    def export_jobs_as_images(self, jobs: list[ImageJob], target: Path) -> tuple[int, list[str]]:
        exported = 0
        missing: list[str] = []
        for job in jobs:
            try:
                destination = self.export_one_job_as_image(job, target)
            except OSError as exc:
                missing.append(f"{job.path.name}: {exc}")
                continue
            if destination is None:
                missing.append(job.path.name)
                continue
            exported += 1
        return exported, missing

    def export_one_job_as_image(self, job: ImageJob, target: Path) -> Path | None:
        suffix = job.path.suffix.lower()
        stem = stm_flatten.safe_stem(job.path)
        if suffix in {".dat", ".tif", ".tiff"}:
            source = self.tiff_export_source(job)
            if source is None:
                return None
            destination = self.unique_export_path(target / f"{stem}_flattened.tiff")
            shutil.copy2(source, destination)
            return destination

        if suffix in {".jpg", ".jpeg"}:
            destination = self.unique_export_path(target / f"{stem}_flattened.jpeg")
            image = self.jpeg_export_image(job)
            if image is None:
                return None
            image.convert("RGB").save(destination, format="JPEG", quality=95, subsampling=0)
            return destination

        return None

    def tiff_export_source(self, job: ImageJob) -> Path | None:
        if job.scientific_output_path and job.scientific_output_path.exists() and job.scientific_output_path.suffix.lower() in {".tif", ".tiff"}:
            return job.scientific_output_path
        path = self.report_output_path(job, "flattened_tiff")
        if path and path.exists():
            return path
        return None

    def jpeg_export_image(self, job: ImageJob) -> Image.Image | None:
        # JPEG inputs have no calibrated scientific height output. Export the
        # flattened image preview as JPEG instead of the internal .npy array.
        path = self.report_output_path(job, "flattened_png")
        if path and path.exists():
            with Image.open(path) as img:
                return img.convert("RGB")
        if job.scientific_output_path and job.scientific_output_path.exists() and job.scientific_output_path.suffix.lower() == ".npy":
            return self.array_to_display_image(np.load(job.scientific_output_path))
        return None

    def unique_export_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        for idx in range(1, 10000):
            candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise OSError(f"Could not find an unused export name for {path.name}")

    def open_selected_diagnostic(self) -> None:
        if not self.selected_indices:
            messagebox.showinfo("No selection", "Select a row first.")
            return
        job = self.jobs[sorted(self.selected_indices)[0]]
        self.open_job_diagnostic(job)

    def open_job_diagnostic(self, job: ImageJob) -> None:
        diagnostic = job.diagnostic_path
        if diagnostic is None and job.report:
            value = job.report.get("outputs", {}).get("diagnostic_png")
            diagnostic = Path(value) if value else None
        if diagnostic is None or not diagnostic.exists():
            messagebox.showinfo("No diagnostic", "Run this image first to create a diagnostic PNG.")
            return
        os.startfile(diagnostic)

    def get_tile(self, job: ImageJob, key: str, size: int) -> ImageTk.PhotoImage | None:
        source_id, mtime, loader = self.tile_source(job, key)
        if loader is None:
            return None
        cache_key = (source_id, key, size, mtime)
        cached = self.image_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            image = loader()
            image = self.fit_tile(image, size)
            photo = ImageTk.PhotoImage(image)
            self.image_cache[cache_key] = photo
            return photo
        except Exception:
            return None

    def tile_source(self, job: ImageJob, key: str) -> tuple[str, float, Any | None]:
        if key == "original":
            return str(job.path), self.safe_mtime(job.path), lambda: self.display_image_from_path(job.path)

        if key == "flattened":
            if job.scientific_output_path and job.scientific_output_path.exists():
                path = job.scientific_output_path
                return str(path), self.safe_mtime(path), lambda: self.display_image_from_scientific(path)
            path = self.report_output_path(job, "flattened_png")
            if path and path.exists():
                return str(path), self.safe_mtime(path), lambda: self.display_png(path)
            return "missing-flattened", 0.0, None

        if key == "fit_regions":
            path = self.report_output_path(job, "fit_mask_png")
            if path and path.exists():
                return str(path), self.safe_mtime(path), lambda: self.display_png(path)
            return self.diagnostic_source(job, "fit_regions")

        if key == "background":
            path = self.report_output_path(job, "background_png")
            if path and path.exists():
                return str(path), self.safe_mtime(path), lambda: self.display_png(path)
            return self.diagnostic_source(job, "background")

        if key == "residual":
            return self.diagnostic_source(job, "residual")

        return "missing", 0.0, None

    def diagnostic_source(self, job: ImageJob, panel: str) -> tuple[str, float, Any | None]:
        path = job.diagnostic_path
        if path is None and job.report:
            value = job.report.get("outputs", {}).get("diagnostic_png")
            path = Path(value) if value else None
        if path is None or not path.exists():
            return f"missing-{panel}", 0.0, None
        return f"{path}:{panel}", self.safe_mtime(path), lambda: self.crop_diagnostic_panel(path, panel)

    def report_output_path(self, job: ImageJob, key: str) -> Path | None:
        if not job.report:
            return None
        value = job.report.get("outputs", {}).get(key)
        return Path(value) if value else None

    def display_image_from_path(self, path: Path) -> Image.Image:
        if path.suffix.lower() == ".dat":
            dat_topography, _meta = stm_flatten.load_dat_forward_topography(path)
            return self.array_to_display_image(dat_topography)
        if path.suffix.lower() in {".tif", ".tiff"}:
            return self.array_to_display_image(tifffile.imread(path))
        with Image.open(path) as img:
            return img.convert("RGB")

    def display_image_from_scientific(self, path: Path) -> Image.Image:
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            return self.array_to_display_image(tifffile.imread(path))
        if suffix == ".npy":
            return self.array_to_display_image(np.load(path))
        with Image.open(path) as img:
            return img.convert("RGB")

    def display_png(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def array_to_display_image(self, arr: np.ndarray) -> Image.Image:
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            rgb = arr[:, :, :3].astype(np.float64, copy=False)
            if np.nanmax(rgb) <= 1.5:
                rgb = rgb * 255.0
            return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[0]
        u8 = stm_flatten.normalize_uint8(arr)
        return Image.fromarray(u8, mode="L").convert("RGB")

    def crop_diagnostic_panel(self, path: Path, panel: str) -> Image.Image:
        # Diagnostic PNGs are 2x3 matplotlib figures. These fractional boxes
        # deliberately crop inside each subplot, removing most title/axis space.
        boxes = {
            "original": (0.02, 0.03, 0.30, 0.47),
            "fit_regions": (0.33, 0.03, 0.61, 0.47),
            "background": (0.70, 0.03, 0.98, 0.47),
            "flattened": (0.02, 0.50, 0.30, 0.94),
            "residual": (0.33, 0.50, 0.61, 0.94),
        }
        box = boxes[panel]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            crop_box = (
                int(box[0] * width),
                int(box[1] * height),
                int(box[2] * width),
                int(box[3] * height),
            )
            return rgb.crop(crop_box)

    def fit_tile(self, image: Image.Image, size: int) -> Image.Image:
        tile = Image.new("RGB", (size, size), (242, 244, 247))
        image = image.copy()
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        tile.paste(image, (x, y))
        return tile

    def clear_image_cache(self) -> None:
        self.image_cache.clear()

    def safe_mtime(self, path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0


def main() -> int:
    app = FlattenGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
