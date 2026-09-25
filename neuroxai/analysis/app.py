"""Main window of the Analysis Tool: a toolbar and a grid of chart panes."""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from tensorflow import keras

from .. import imaging, xai
from ..config import RESEARCH_NOTICE
from ..model_check import check_model
from ..ui import maximize_window
from . import charts
from .layer_dialog import LayerDialog
from .panes import ChartPane
from .pipeline import analyze, summary_text

IMAGE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
MODEL_TYPES = [("Keras models", "*.keras *.h5"), ("All files", "*.*")]
READY = "Ready: open a coronal slice"


class AnalysisApp:
    """The Analysis Tool. It classifies one coronal slice and shows the explanation maps in 16 panes."""

    def __init__(self, root, model_path=None, image_path=None):
        self.root = root
        self.model = self.model_info = self.model_path = self.explainer = None
        self.analysis = None
        self.maximized_pane = None
        self._saved_sashes = None

        root.title("NeuroXAI Analysis Tool")
        root.geometry("1400x1000")
        maximize_window(root)
        self._build_toolbar()
        ttk.Label(root, text=RESEARCH_NOTICE, foreground="#586e75").pack(
            side=tk.BOTTOM, anchor=tk.W, padx=10, pady=(0, 4)
        )
        self._build_grid()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(200, lambda: self._start(model_path, image_path))

    # Layout

    def _build_toolbar(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bar, text="Open Slice", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(bar, text="Load Model", command=self.open_model).pack(side=tk.LEFT, padx=5)
        self.layer_button = ttk.Button(bar, text="Explanation Layers", command=self.configure_layers, state=tk.DISABLED)
        self.layer_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(bar, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(bar, text="Save Report", command=self.save_report).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(bar, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT, padx=15)
        self.status = tk.StringVar(value="Starting...")
        ttk.Label(bar, textvariable=self.status, width=48, anchor=tk.E).pack(side=tk.RIGHT, padx=5)

    def _build_grid(self):
        self.grid_frame = ttk.Frame(self.root)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.rows = ttk.PanedWindow(self.grid_frame, orient=tk.VERTICAL)
        self.rows.pack(fill=tk.BOTH, expand=True)
        self.row_panes = []
        self.panes = {}
        for index, (title, placeholder) in enumerate(charts.PANES):
            if index % charts.COLUMNS == 0:
                row = ttk.PanedWindow(self.rows, orient=tk.HORIZONTAL)
                self.rows.add(row, weight=1)
                self.row_panes.append(row)
            pane = ChartPane(self.row_panes[-1], title, self.toggle_maximize, placeholder)
            self.row_panes[-1].add(pane, weight=1)
            self.panes[title] = pane

    def set_status(self, message, progress=None):
        self.status.set(message)
        if progress is not None:
            self.progress["value"] = progress
        self.root.update_idletasks()

    # Model

    def _start(self, model_path, image_path):
        if model_path and Path(model_path).exists():
            self.load_model(model_path, interactive=False)
        else:
            self.set_status("No model: click Load Model", 0)
        if image_path and self.model is not None:
            self.analyze_file(image_path)

    def open_model(self):
        path = filedialog.askopenfilename(title="Select a Keras model", filetypes=MODEL_TYPES, parent=self.root)
        if path:
            self.load_model(path)

    def load_model(self, path, interactive=True):
        """Load and check a model. In interactive mode, ask before a model with warnings is used."""
        path = Path(path)
        self.set_status(f"Loading {path.name}...", 20)
        try:
            model = keras.models.load_model(path)
            info = check_model(model)
        except Exception as e:  # Keras raises many types of errors for files that it cannot read
            self.set_status("The model did not load", 0)
            messagebox.showerror("Load Model", f"Could not load the model:\n{e}", parent=self.root)
            return False
        if info.errors:
            self.set_status("The model cannot be used", 0)
            messagebox.showerror(
                "Load Model", "NeuroXAI cannot use this model:\n\n" + "\n\n".join(info.errors), parent=self.root
            )
            return False
        if interactive and info.warnings:
            question = "\n\n".join(info.warnings) + "\n\nLoad the model?"
            if not messagebox.askyesno("Load Model", question, icon="warning", parent=self.root):
                self.set_status("Model not loaded", 0)
                return False
        self.model, self.model_info, self.model_path = model, info, path
        self.explainer = xai.Explainer(model)
        self.layer_button.config(state=tk.NORMAL)
        self.set_status(f"Model loaded: {path.name}", 100)
        if interactive:
            messagebox.showinfo("Model Information", f"Model: {path.name}\n\n{info.describe()}", parent=self.root)
        self.root.after(1500, lambda: self.set_status(READY, 0))
        return True

    def configure_layers(self):
        if self.explainer is None:
            messagebox.showwarning("Explanation Layers", "Load a model first.", parent=self.root)
            return
        result = LayerDialog(self.root, self.explainer).show()
        if result is None:
            return
        if result["reanalyze"] and self.analysis is not None:
            self._analyze(self.analysis.image, self.analysis.image_path)
        else:
            self.set_status("Layers changed. They apply to the next analysis.")

    # Analysis

    def open_image(self):
        if self.model is None:
            messagebox.showwarning("Open Slice", "Load a model first.", parent=self.root)
            return
        path = filedialog.askopenfilename(title="Select a coronal MRI slice", filetypes=IMAGE_TYPES, parent=self.root)
        if path:
            self.analyze_file(path)

    def analyze_file(self, path):
        try:
            image = imaging.load_slice(path)
        except OSError as e:
            messagebox.showerror("Open Slice", f"Could not read the image:\n{e}", parent=self.root)
            return
        self._analyze(image, path)

    def _analyze(self, image, path):
        self.set_status("Classifying the slice and making the maps...", 30)
        try:
            self.analysis = analyze(
                self.model, self.explainer, self.model_info.preprocess, image, path, self.model_path.name
            )
            self.set_status("Drawing the charts...", 80)
            figures = charts.build_all(self.analysis)
        except Exception as e:  # show any failure of TensorFlow or matplotlib to the user
            self.set_status("The analysis failed", 0)
            messagebox.showerror("Analysis", f"The analysis failed:\n{e}", parent=self.root)
            return
        for title, pane in self.panes.items():
            pane.show(figures[title])
        layers = ", ".join(f"{xai.METHODS[m]}: {layer}" for m, layer in self.analysis.layers.items())
        self.set_status(f"{self.analysis.label} ({self.analysis.confidence:.1%}). {layers}", 100)

    def clear(self):
        self._restore_layout()
        for pane in self.panes.values():
            pane.clear()
        self.analysis = None
        self.set_status(READY if self.model is not None else "No model: click Load Model", 0)

    def save_report(self):
        """Save all figures (300 dpi) and a text and JSON summary in a new folder."""
        if self.analysis is None:
            messagebox.showinfo("Save Report", "There is no analysis to save.", parent=self.root)
            return
        folder = filedialog.askdirectory(title="Select a folder for the report", parent=self.root)
        if not folder:
            return
        report_dir = Path(folder) / f"neuroxai_report_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            (report_dir / "figures").mkdir(parents=True)
            for i, (title, pane) in enumerate(self.panes.items()):
                self.set_status(f"Saving {title}...", 100 * i / len(self.panes))
                name = title.replace(" ", "_").replace("+", "Plus")
                pane.figure.savefig(
                    report_dir / "figures" / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white"
                )
            (report_dir / "summary.txt").write_text(summary_text(self.analysis) + "\n")
            (report_dir / "summary.json").write_text(json.dumps(self.analysis.to_dict(), indent=2) + "\n")
        except OSError as e:
            self.set_status("The report was not saved", 0)
            messagebox.showerror("Save Report", f"Could not save the report:\n{e}", parent=self.root)
            return
        self.set_status("Report saved", 100)
        messagebox.showinfo("Save Report", f"The report is in:\n{report_dir}", parent=self.root)

    # Maximizing one pane

    def toggle_maximize(self, pane):
        if self.maximized_pane is pane:
            self._restore_layout()
        else:
            self._restore_layout()
            self._maximize(pane)

    def _maximize(self, pane):
        names = [[str(p) for p in row.panes()] for row in self.row_panes]
        row = next(i for i, row_names in enumerate(names) if str(pane) in row_names)
        column = names[row].index(str(pane))
        self.root.update_idletasks()
        self._saved_sashes = self._sash_positions()
        height, width = self.grid_frame.winfo_height(), self.grid_frame.winfo_width()
        # Move the sashes before the pane to the start and the sashes after it to the end.
        for i in range(len(self.row_panes) - 1):
            self.rows.sashpos(i, 1 if i < row else height - 1)
        for j in range(charts.COLUMNS - 1):
            self.row_panes[row].sashpos(j, 1 if j < column else width - 1)
        self.maximized_pane = pane
        pane.set_maximized(True)

    def _restore_layout(self):
        if self.maximized_pane is None:
            return
        self.maximized_pane.set_maximized(False)
        self.maximized_pane = None
        rows, columns = self._saved_sashes
        for i, position in enumerate(rows):
            self.rows.sashpos(i, position)
        for row, positions in zip(self.row_panes, columns, strict=True):
            for j, position in enumerate(positions):
                row.sashpos(j, position)

    def _sash_positions(self):
        rows = [self.rows.sashpos(i) for i in range(len(self.row_panes) - 1)]
        columns = [[row.sashpos(j) for j in range(len(row.panes()) - 1)] for row in self.row_panes]
        return rows, columns

    def reset_view(self):
        """Restore a maximized pane and give all rows and columns the same size."""
        self._restore_layout()
        self.root.update_idletasks()
        height, width = self.grid_frame.winfo_height(), self.grid_frame.winfo_width()
        for i in range(len(self.row_panes) - 1):
            self.rows.sashpos(i, (i + 1) * height // len(self.row_panes))
        for row in self.row_panes:
            for j in range(charts.COLUMNS - 1):
                row.sashpos(j, (j + 1) * width // charts.COLUMNS)
        self.set_status("View reset")

    def on_close(self):
        self.root.destroy()
