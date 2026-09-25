"""Volumes, sessions, reports and patient information of the Slice Viewer."""

import importlib.util
import json
import re
import subprocess
import sys
import tkinter as tk
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .. import session
from ..annotations import DRAWING, MEASUREMENT, AnnotationStore, summary
from ..config import REPO_ROOT
from ..volume import FILE_TYPES, Volume

SESSION_TYPES = [("Session files", "*.json"), ("All files", "*.*")]


def _safe(text, default):
    """Text with only letters and digits, for file names."""
    return re.sub(r"[\W_]+", "", text) or default


class FilesMixin:
    """Loading and saving. Each operation shows its errors in a dialog."""

    def set_status(self, message):
        self.status_label.config(text=message)

    @contextmanager
    def _busy(self, message):
        self.set_status(message)
        self.progress.pack(side=tk.RIGHT, padx=5)
        self.progress.start()
        self.root.update_idletasks()
        try:
            yield
        finally:
            self.progress.stop()
            self.progress.pack_forget()

    # Volumes

    def open_volume(self, path=None):
        path = path or filedialog.askopenfilename(title="Select a volume", filetypes=FILE_TYPES, parent=self.root)
        if not path:
            return False
        try:
            with self._busy(f"Loading {Path(path).name}..."):
                volume = Volume.load(path)
        except (RuntimeError, OSError) as e:  # SimpleITK raises RuntimeError for files that it cannot read
            messagebox.showerror("Load Volume", f"Could not load the volume:\n{e}", parent=self.root)
            self.set_status("Loading failed")
            return False
        self._set_volume(volume)
        self.patient = {"name": "N/A", "id": "N/A"}
        self._show_patient()
        self.set_status(f"Loaded {Path(path).name}: {' x '.join(map(str, reversed(volume.shape)))} voxels")
        return True

    def _set_volume(self, volume):
        """Show a new volume with default settings and no annotations."""
        self.volume = volume
        self.store = AnnotationStore()
        self.pending_distance = self.pending_area = self.zoom_rect = self.highlight = None
        low, high = volume.value_range()
        self.level = (low + high) / 2
        self.window = max(high - low, 1.0)
        self.window_slider.config(from_=1, to=2 * self.window)
        self.level_slider.config(from_=low, to=high)
        for plane, slider in self.slice_sliders.items():
            slider.config(from_=0, to=volume.shape[plane] - 1)
        self.current_slices = [size // 2 for size in volume.shape]
        for name in self.views:
            self.zoom[name] = 1.0
            self.zoom_center[name] = (0.5, 0.5)
        self._sync_controls()
        self._after_annotation_change()

    # Patient information

    def _show_patient(self):
        self.patient_label.config(text=f"Patient: {self.patient['name']} (ID: {self.patient['id']})")

    def edit_patient_info(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Patient Information")
        dialog.transient(self.root)
        dialog.configure(bg=self.style.lookup("TFrame", "background"))
        dialog.geometry(f"+{self.root.winfo_rootx() + 50}+{self.root.winfo_rooty() + 50}")
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(expand=True, fill=tk.BOTH)
        ttk.Label(frame, text="Patient Information", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, pady=(0, 15), sticky=tk.W
        )
        values = {}
        for row, (key, label) in enumerate((("name", "Patient Name:"), ("id", "Patient ID:")), start=1):
            ttk.Label(frame, text=label).grid(row=row, column=0, padx=(0, 10), pady=5, sticky=tk.W)
            current = self.patient[key]
            values[key] = tk.StringVar(value="" if current == "N/A" else current)
            entry = ttk.Entry(frame, textvariable=values[key], width=25)
            entry.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)
            if row == 1:
                entry.focus_set()
        frame.columnconfigure(1, weight=1)

        def accept():
            self.patient = {key: var.get().strip() or "N/A" for key, var in values.items()}
            self._show_patient()
            dialog.destroy()
            self.set_status("Patient information changed")

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, pady=(15, 0))
        ttk.Button(buttons, text="OK", command=accept).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=(5, 0))
        dialog.bind("<Return>", lambda _: accept())
        dialog.bind("<Escape>", lambda _: dialog.destroy())
        dialog.grab_set()
        dialog.wait_window()

    # Sessions

    def save_session(self):
        if self.volume is None:
            messagebox.showerror("Save Session", "Load a volume first.", parent=self.root)
            return
        patient = f"{_safe(self.patient['id'], 'NA')}_{_safe(self.patient['name'], 'Patient')}"
        name = f"Session_{patient}_{datetime.now():%Y%m%d_%H%M%S}.json"
        path = filedialog.asksaveasfilename(
            initialfile=name, defaultextension=".json", filetypes=SESSION_TYPES, title="Save Session", parent=self.root
        )
        if not path:
            return
        view_state = {
            "current_slices": list(self.current_slices),
            "window": self.window,
            "level": self.level,
            "zoom_factors": dict(self.zoom),
            "zoom_centers": dict(self.zoom_center),
            "colormap": self.colormap.get(),
            "crosshair_enabled": self.crosshair_enabled.get(),
            "show_overlays": self.show_overlays.get(),
        }
        tool_state = {
            "draw_color": self.draw_color.get(),
            "draw_size": self.draw_size.get(),
            "draw_opacity": self.draw_opacity.get(),
        }
        try:
            with self._busy("Saving the session..."):
                session.write(path, session.build(self.volume, self.store, self.patient, view_state, tool_state))
        except OSError as e:
            messagebox.showerror("Save Session", f"Could not save the session:\n{e}", parent=self.root)
            return
        self.set_status(f"Session saved: {Path(path).name}")

    def load_session(self):
        path = filedialog.askopenfilename(filetypes=SESSION_TYPES, title="Load Session", parent=self.root)
        if not path:
            return
        try:
            data = session.read(path)
        except (OSError, ValueError) as e:  # json.JSONDecodeError is a ValueError
            messagebox.showerror("Load Session", f"Could not read the session:\n{e}", parent=self.root)
            return
        volume_file = data["session_info"].get("volume_file", "")
        if not Path(volume_file).exists():
            if not messagebox.askyesno(
                "Load Session",
                f"The volume of the session was not found:\n{volume_file}\n\nSelect the volume file?",
                parent=self.root,
            ):
                return
            volume_file = filedialog.askopenfilename(title="Select the volume", filetypes=FILE_TYPES, parent=self.root)
            if not volume_file:
                return
        if not self.open_volume(volume_file):
            return
        if not session.checksum_matches(data, self.volume) and not messagebox.askyesno(
            "Load Session",
            "The volume is not the same as the volume of the session. "
            "The annotations can be in the wrong positions.\n\nContinue?",
            parent=self.root,
        ):
            return
        if session.is_legacy(data):
            messagebox.showwarning(
                "Load Session",
                "This session is from NeuroXAI 1.0 or 1.1. These versions did not reorient the volume. "
                "If the volume file is not in LPS orientation, the annotations are in the wrong positions.",
                parent=self.root,
            )
        self._apply_session(data)
        self.set_status(f"Session loaded: {Path(path).name}")

    def _apply_session(self, data):
        patient = data.get("patient_info", {})
        self.patient = {"name": patient.get("name", "N/A"), "id": patient.get("id", "N/A")}
        self._show_patient()
        view_state = data["view_state"]
        self.current_slices = [
            int(min(max(index, 0), size - 1))
            for index, size in zip(view_state["current_slices"], self.volume.shape, strict=True)
        ]
        self.window = float(view_state["window"])
        self.level = float(view_state["level"])
        for name in self.views:
            self.zoom[name] = float(view_state.get("zoom_factors", {}).get(name, 1.0))
            self.zoom_center[name] = tuple(view_state.get("zoom_centers", {}).get(name, (0.5, 0.5)))
        self.colormap.set(view_state.get("colormap", "gray"))
        self.crosshair_enabled.set(view_state.get("crosshair_enabled", True))
        self.show_overlays.set(view_state.get("show_overlays", True))
        tool_state = data.get("tool_state", {})
        self.draw_color.set(tool_state.get("draw_color", "yellow"))
        self.draw_size.set(tool_state.get("draw_size", 2.0))
        self.draw_opacity.set(tool_state.get("draw_opacity", 0.8))
        self.store = session.annotations(data)
        self._sync_controls()
        self._after_annotation_change()

    # Reports

    def save_report(self):
        """Save report.json and a snapshot of each annotation that has a comment."""
        if self.volume is None:
            messagebox.showerror("Save Report", "Load a volume first.", parent=self.root)
            return
        folder = filedialog.askdirectory(title="Select a folder for the report", parent=self.root)
        if not folder:
            return
        report_dir = Path(folder) / f"Report_{_safe(self.patient['id'], 'NA')}_{datetime.now():%Y%m%d_%H%M%S}"
        report = {
            "patient_info": self.patient,
            "source_file": self.volume.path,
            "report_date": datetime.now().isoformat(timespec="seconds"),
            "view_settings": {
                "window": self.window,
                "level": self.level,
                "current_slices": self.current_slices,
                "zoom_factors": self.zoom,
            },
            "drawings": [],
            "measurements": [],
        }
        try:
            with self._busy("Saving the report..."):
                (report_dir / "snapshots").mkdir(parents=True)
                for kind, key, prefix in (
                    (DRAWING, "drawings", "drawing"),
                    (MEASUREMENT, "measurements", "measurement"),
                ):
                    for item in self.store.items(kind):
                        if not item.get("comment"):
                            continue
                        image_file = Path("snapshots") / f"{prefix}_{item['id']}.png"
                        self._save_snapshot(kind, item, report_dir / image_file)
                        report[key].append({**item, "summary": summary(kind, item), "image_file": str(image_file)})
                with open(report_dir / "report.json", "w") as f:
                    json.dump(report, f, indent=2, cls=session.NumpyEncoder)
        except OSError as e:
            messagebox.showerror("Save Report", f"Could not save the report:\n{e}", parent=self.root)
            return
        count = len(report["drawings"]) + len(report["measurements"])
        self.set_status(f"Report saved with {count} commented annotations: {report_dir}")
        messagebox.showinfo("Save Report", f"The report is in:\n{report_dir}", parent=self.root)

    # Other windows

    def launch_analysis_tool(self):
        """Start the Analysis Tool in its own process, because it needs TensorFlow."""
        missing = [name for name in ("tensorflow", "cv2") if importlib.util.find_spec(name) is None]
        if missing:
            messagebox.showerror(
                "Analysis Tool", f"The Analysis Tool needs these packages: {', '.join(missing)}.", parent=self.root
            )
            return
        subprocess.Popen([sys.executable, "-m", "neuroxai.analysis"], cwd=REPO_ROOT)
        self.set_status("Analysis Tool started")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Quit the Slice Viewer?", parent=self.root):
            self.root.destroy()
