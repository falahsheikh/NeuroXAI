"""Mouse, keyboard and slider input of the Slice Viewer.

Without a tool mode: the left button moves the crosshair, the middle button pans,
the right button changes the window and level, and the wheel changes the slice (with Ctrl: zoom).
In a tool mode, the left button measures, draws, zooms to a rectangle or pans.
"""

import math

import numpy as np

from ..geometry import format_intensity
from ..volume import IN_PLANE_AXES, VIEWS

# Cursor and status message for each tool mode. Only one mode can be on.
MODES = {
    "measure": ("cross", "Measure: click the start point, then the end point. Esc cancels."),
    "area": ("crosshair", "Area: drag around a region. The outline closes when you release the button."),
    "draw": ("cross", "Draw: drag on a view to draw."),
    "zoom_select": ("crosshair", "Zoom Select: drag a rectangle on a view to zoom to it."),
    "pan": ("fleur", "Pan: drag to move the view."),
}
ZOOM_STEP = 1.2
MAX_ZOOM = 100.0
AREA_MIN_STEP = 2.0  # pixels between the points of an area outline


class InteractionMixin:
    """Event handlers. They change the state of the viewer and then draw the views again."""

    # Window and level

    @property
    def vmin(self):
        return self.level - self.window / 2

    @property
    def vmax(self):
        return self.level + self.window / 2

    def _on_window_slider(self, value):
        if not self._syncing:
            self.window = max(1.0, float(value))
            self._sync_controls()
            self.update_views()

    def _on_level_slider(self, value):
        if not self._syncing:
            self.level = float(value)
            self._sync_controls()
            self.update_views()

    def _on_slice_slider(self, plane, value):
        if not self._syncing and self.volume is not None:
            self.current_slices[plane] = int(round(float(value)))
            self._sync_controls()
            self.update_views()

    def _sync_controls(self):
        """Show the current window, level and slice numbers on the sliders and labels."""
        self._syncing = True  # a Scale calls its command when set() changes it
        try:
            self.window_slider.set(self.window)
            self.level_slider.set(self.level)
            self.window_value.config(text=format_intensity(self.window))
            self.level_value.config(text=format_intensity(self.level))
            for plane, slider in self.slice_sliders.items():
                if self.volume is None:
                    self.slice_values[plane].config(text="0/0")
                    continue
                slider.set(self.current_slices[plane])
                self.slice_values[plane].config(text=f"{self.current_slices[plane]}/{self.volume.shape[plane] - 1}")
        finally:
            self._syncing = False

    # Tool modes

    def active_mode(self):
        return next((name for name, var in self.modes.items() if var.get()), None)

    def _on_mode_toggle(self, name):
        if self.modes[name].get():
            for other, var in self.modes.items():
                if other != name and var.get():
                    var.set(False)
            cursor, message = MODES[name]
            self._set_cursor(cursor)
            self.set_status(message)
        else:
            self._cancel_pending_of(name)
            if self.active_mode() is None:
                self._set_cursor("")
                self.set_status("Ready")
        self.update_views()

    def _set_cursor(self, cursor):
        for view in self.views.values():
            view["canvas"].get_tk_widget().config(cursor=cursor)

    def _cancel_pending_of(self, mode):
        if mode == "measure":
            self.pending_distance = None
        elif mode == "area":
            self.pending_area = None
        elif mode == "zoom_select":
            self.zoom_rect = None

    def cancel_pending(self):
        """Cancel a measurement, an area outline or a zoom rectangle that is not complete (Esc)."""
        if self.pending_distance or self.pending_area or self.zoom_rect:
            self.pending_distance = self.pending_area = self.zoom_rect = None
            self.drag = None
            self.set_status("Canceled")
            self.update_views()

    # Mouse

    def _on_press(self, event, view):
        if self.volume is None or event.inaxes is None or event.xdata is None:
            return
        plane = VIEWS[view]
        index = self.current_slices[plane]
        point = (event.xdata, event.ydata)
        self.last_mouse = (event.x, event.y)
        mode = self.active_mode() if event.button == 1 else None
        if mode == "area":
            self.pending_area = {"plane": plane, "slice": index, "points": [point]}
            self.drag = "area"
        elif mode == "draw":
            self.active_stroke = self.store.begin_stroke(
                plane, index, point, self.draw_color.get(), float(self.draw_size.get()), float(self.draw_opacity.get())
            )
            self.drag = "draw"
            self._after_annotation_change()
        elif mode == "measure":
            self._measure_click(plane, index, point)
        elif mode == "zoom_select":
            self.zoom_rect = {"view": view, "start": point, "end": point}
            self.drag = "zoom_select"
        elif mode == "pan" or event.button == 2:
            self.drag = "pan"
        elif event.button == 1:
            self.drag = "crosshair"
            self._move_crosshair(view, point)
        elif event.button == 3:
            self.drag = "window_level"
        self.update_views()

    def _on_motion(self, event, view):
        self._show_position(event, view)
        if self.drag is None or event.inaxes is None or event.xdata is None:
            return
        point = (event.xdata, event.ydata)
        if self.drag == "draw":
            if self.store.extend_stroke(self.active_stroke, point):
                self._draw_view(view)
        elif self.drag == "area":
            if math.dist(self.pending_area["points"][-1], point) > AREA_MIN_STEP:
                self.pending_area["points"].append(point)
                self._draw_view(view)
        elif self.drag == "zoom_select":
            self.zoom_rect["end"] = point
            self._draw_view(view)
        elif self.drag == "pan":
            self._pan(view, event)
        elif self.drag == "crosshair":
            self._move_crosshair(view, point)
        elif self.drag == "window_level":
            dx, dy = event.x - self.last_mouse[0], event.y - self.last_mouse[1]
            low, high = self.volume.value_range()
            self.window = max(1.0, self.window + dx * (high - low) / 100)
            self.level -= dy * (high - low) / 200
            self._sync_controls()
            self.update_views()
        self.last_mouse = (event.x, event.y)

    def _on_release(self, event, view):
        if self.drag == "area":
            self._finish_area()
        elif self.drag == "zoom_select":
            self._finish_zoom_select(view)
        self.drag = None
        self.active_stroke = None

    def _on_scroll(self, event, view):
        if self.volume is None or event.xdata is None:
            return
        plane = VIEWS[view]
        if event.key == "control":
            factor = ZOOM_STEP if event.button == "up" else 1 / ZOOM_STEP
            self.zoom[view] = float(np.clip(self.zoom[view] * factor, 1.0, MAX_ZOOM))
            height, width = self.volume.slice(plane, self.current_slices[plane]).shape
            self.zoom_center[view] = (event.xdata / width, event.ydata / height)
        else:
            step = -1 if event.button == "up" else 1
            self.current_slices[plane] = int(
                np.clip(self.current_slices[plane] + step, 0, self.volume.shape[plane] - 1)
            )
            self._sync_controls()
        self.update_views()

    def _on_view_enter(self, event, view):
        self.view_label.config(text=f"View: {view.capitalize()}")

    def _on_view_leave(self, event):
        self.view_label.config(text="View: ---")
        self._show_position(None, None)

    # Actions

    def _move_crosshair(self, view, point):
        """Set the slices of the other two planes to the voxel under point."""
        for plane, value in zip(IN_PLANE_AXES[VIEWS[view]], point, strict=True):
            self.current_slices[plane] = self._voxel_index(plane, value)
        self._sync_controls()
        self.update_views()

    def _voxel_index(self, plane, value):
        """Index along the axis of plane of the voxel that contains the data coordinate value."""
        return int(np.clip(math.floor(value), 0, self.volume.shape[plane] - 1))

    def _measure_click(self, plane, index, point):
        pending = self.pending_distance
        if pending is None or (pending["plane"], pending["slice"]) != (plane, index):
            self.pending_distance = {"plane": plane, "slice": index, "p1": point}
            self.set_status("Measure: click the end point. Esc cancels.")
            return
        measurement = self.store.add_distance(
            plane, index, pending["p1"], point, self.draw_color.get(), self.volume.pixel_spacing(plane)
        )
        self.pending_distance = None
        self._after_annotation_change()
        self.set_status(f"Distance: {measurement['dist_mm']:.2f} mm")

    def _finish_area(self):
        area, self.pending_area = self.pending_area, None
        if area is None or len(area["points"]) < 3:
            self.set_status("Area canceled: drag around a region to make an outline.")
            self.update_views()
            return
        measurement = self.store.add_area(
            area["plane"],
            area["slice"],
            area["points"],
            self.draw_color.get(),
            self.volume.pixel_spacing(area["plane"]),
        )
        self._after_annotation_change()
        self.set_status(f"Area: {measurement['area_mm2']:.2f} mm²")

    def _finish_zoom_select(self, view):
        rect, self.zoom_rect = self.zoom_rect, None
        (x0, y0), (x1, y1) = rect["start"], rect["end"]
        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            self.update_views()
            return
        plane = VIEWS[view]
        height, width = self.volume.slice(plane, self.current_slices[plane]).shape
        ax = self.views[view]["ax"]
        shown_width = abs(np.diff(ax.get_xlim())[0])
        shown_height = abs(np.diff(ax.get_ylim())[0])
        factor = min(shown_width / abs(x1 - x0), shown_height / abs(y1 - y0))
        self.zoom[view] = float(np.clip(self.zoom[view] * factor, 1.0, MAX_ZOOM))
        self.zoom_center[view] = ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height)
        self.update_views()

    def _pan(self, view, event):
        """Move the view so that the image point under the mouse follows the mouse."""
        ax = self.views[view]["ax"]
        to_data = ax.transData.inverted()
        (x0, y0), (x1, y1) = to_data.transform([self.last_mouse, (event.x, event.y)])
        plane = VIEWS[view]
        height, width = self.volume.slice(plane, self.current_slices[plane]).shape
        cx, cy = self.zoom_center[view]
        self.zoom_center[view] = (
            float(np.clip(cx - (x1 - x0) / width, 0, 1)),
            float(np.clip(cy - (y1 - y0) / height, 0, 1)),
        )
        self._draw_view(view)

    def reset_zoom(self):
        for name in self.views:
            self.zoom[name] = 1.0
            self.zoom_center[name] = (0.5, 0.5)
        self.update_views()

    def _show_position(self, event, view):
        """Show the voxel under the mouse and its intensity in the status bar."""
        if self.volume is None or event is None or event.inaxes is None or event.xdata is None:
            self.position_label.config(text="Voxel: ---")
            self.intensity_label.config(text="Intensity: ---")
            return
        position = list(self.current_slices)
        for plane, value in zip(IN_PLANE_AXES[VIEWS[view]], (event.xdata, event.ydata), strict=True):
            position[plane] = self._voxel_index(plane, value)
        z, y, x = position
        self.position_label.config(text=f"Voxel (x, y, z): {x}, {y}, {z}")
        self.intensity_label.config(text=f"Intensity: {self.volume.data[z, y, x]:.2f}")
