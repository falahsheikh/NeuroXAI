"""Main window of the Slice Viewer."""

import tkinter as tk

from ..annotations import AnnotationStore
from ..ui import maximize_window
from ..volume import VIEWS
from .annotation_panel import AnnotationPanelMixin
from .files import FilesMixin
from .interaction import MODES, InteractionMixin
from .layout import LayoutMixin
from .rendering import RenderingMixin


class SliceViewer(LayoutMixin, RenderingMixin, InteractionMixin, AnnotationPanelMixin, FilesMixin):
    """Axial, coronal and sagittal views of an MRI volume, with measurement and annotation tools.

    The window has five parts, one for each mixin:
    LayoutMixin (widgets), RenderingMixin (drawing of the views), InteractionMixin (mouse and sliders),
    AnnotationPanelMixin (annotation lists, undo and redo) and FilesMixin (volumes, sessions and reports).
    """

    def __init__(self, root):
        self.root = root
        root.title("NeuroXAI Slice Viewer")
        root.geometry("1400x1000")

        # Data
        self.volume = None
        self.store = AnnotationStore()
        self.patient = {"name": "N/A", "id": "N/A"}

        # View state
        self.current_slices = [0, 0, 0]  # slice number in each plane (AXIAL, CORONAL, SAGITTAL)
        self.window, self.level = 2000.0, 1000.0
        self.zoom = dict.fromkeys(VIEWS, 1.0)
        self.zoom_center = dict.fromkeys(VIEWS, (0.5, 0.5))  # view center as a fraction of the slice size
        self.maximized_view = None
        self.highlight = None  # (kind, id) of the annotation that is highlighted

        # Settings that widgets show
        self.colormap = tk.StringVar(value="gray")
        self.crosshair_enabled = tk.BooleanVar(value=True)
        self.show_overlays = tk.BooleanVar(value=True)
        self.show_axis_scales = tk.BooleanVar(value=True)
        self.draw_color = tk.StringVar(value="yellow")
        self.draw_size = tk.DoubleVar(value=2.0)
        self.draw_opacity = tk.DoubleVar(value=0.8)
        self.modes = {name: tk.BooleanVar(value=False) for name in MODES}

        # Input that is not complete
        self.drag = None  # what a drag with a mouse button down changes
        self.last_mouse = None  # last mouse position in display pixels
        self.active_stroke = None
        self.pending_distance = None  # {"plane", "slice", "p1"} after the first click of a measurement
        self.pending_area = None  # {"plane", "slice", "points"} while an area outline is drawn
        self.zoom_rect = None  # {"view", "start", "end"} while a zoom rectangle is drawn
        self._syncing = False

        self._build_ui()
        for name, var in self.modes.items():
            var.trace_add("write", lambda *_, name=name: self._on_mode_toggle(name))
        for var in (self.crosshair_enabled, self.show_overlays, self.show_axis_scales):
            var.trace_add("write", lambda *_: self.update_views())
        self._sync_controls()
        self._update_history_buttons()
        self.update_views()
        maximize_window(root)
        root.after(100, self.reset_layout)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
