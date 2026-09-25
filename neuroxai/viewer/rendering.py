"""Drawing of the three views, of the annotations and of the report snapshots."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..annotations import DRAWING, MEASUREMENT
from ..geometry import format_mm, ticks, zoom_limits
from ..volume import AXIS_LABELS, FLIPPED_VERTICAL, IN_PLANE_AXES, VIEWS
from .layout import VIEW_COLORS

PLANE_VIEWS = {plane: name for name, plane in VIEWS.items()}
TEXT_BOX = {"facecolor": "black", "alpha": 0.6, "edgecolor": "none", "pad": 2}


def _hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _label(ax, x, y, text, color, highlighted):
    ax.text(
        x,
        y,
        text,
        color="black" if highlighted else color,
        fontsize=7 if highlighted else 6,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white" if highlighted else "black",
            "alpha": 0.9 if highlighted else 0.7,
            "edgecolor": color,
            "linewidth": 0.5,
        },
    )


def draw_stroke(ax, stroke, highlighted=False):
    points = np.asarray(stroke["points"])
    if len(points) < 2:
        return
    style = {"solid_capstyle": "round", "solid_joinstyle": "round"}
    if highlighted:
        ax.plot(points[:, 0], points[:, 1], color="white", linewidth=stroke["size"] + 4, alpha=0.9, **style)
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=stroke["color"],
        linewidth=stroke["size"],
        alpha=stroke.get("opacity", 0.8),
        **style,
    )


def draw_measurement(ax, measurement, highlighted=False):
    color = measurement.get("color", "yellow")
    width = 2.5 if highlighted else 1.2
    if measurement.get("type") == "area":
        points = np.asarray(measurement["points"])
        outline = np.vstack([points, points[:1]])
        ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=width, alpha=0.9)
        ax.fill(points[:, 0], points[:, 1], color=color, alpha=0.3 if highlighted else 0.15)
        x, y = points.mean(axis=0)
        _label(ax, x, y, f"{measurement['area_mm2']:.1f} mm²", color, highlighted)
    else:
        (x1, y1), (x2, y2) = measurement["p1"], measurement["p2"]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.9)
        ax.plot([x1, x2], [y1, y2], "o", color=color, markersize=6 if highlighted else 4, alpha=0.9)
        _label(ax, (x1 + x2) / 2, (y1 + y2) / 2, f"{measurement['dist_mm']:.1f} mm", color, highlighted)


def draw_start_marker(ax, point):
    ax.plot(point[0], point[1], "o", color="yellow", markersize=6, alpha=0.9)
    ax.text(
        point[0], point[1], "Start", color="yellow", fontsize=6, ha="center", va="bottom", bbox={**TEXT_BOX, "pad": 1}
    )


class RenderingMixin:
    """Draws the views from the state of the viewer."""

    def update_views(self):
        for name in self.views:
            self._draw_view(name)

    def _draw_view(self, name):
        view = self.views[name]
        ax, figure = view["ax"], view["figure"]
        ax.clear()
        ax.set_facecolor("black")
        show_scales = self.volume is not None and self.maximized_view == name and self.show_axis_scales.get()
        if show_scales:
            figure.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.98)
        else:
            figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if self.volume is None:
            ax.text(0.5, 0.5, "No volume loaded", color="grey", ha="center", va="center", transform=ax.transAxes)
            _hide_axes(ax)
            view["canvas"].draw_idle()
            return

        plane = VIEWS[name]
        index = self.current_slices[plane]
        image = self.volume.slice(plane, index)
        height, width = image.shape
        # Pixel (row r, column c) covers [c, c + 1] x [r, r + 1] in data coordinates.
        ax.imshow(
            image,
            cmap=self.colormap.get(),
            vmin=self.vmin,
            vmax=self.vmax,
            origin="lower",
            extent=(0, width, 0, height),
            interpolation="bilinear",
        )
        ax.set_aspect(self.volume.aspect(plane))
        x_limits = zoom_limits(width, self.zoom[name], self.zoom_center[name][0])
        y_limits = zoom_limits(height, self.zoom[name], self.zoom_center[name][1])
        ax.set_xlim(*x_limits)
        ax.set_ylim(*(y_limits[::-1] if plane in FLIPPED_VERTICAL else y_limits))
        if show_scales:
            self._draw_scales(ax, name, plane, x_limits, y_limits)
        else:
            _hide_axes(ax)

        if self.crosshair_enabled.get():
            horizontal, vertical = IN_PLANE_AXES[plane]
            ax.axvline(
                self.current_slices[horizontal] + 0.5, color=VIEW_COLORS[PLANE_VIEWS[horizontal]], lw=0.7, alpha=0.9
            )
            ax.axhline(self.current_slices[vertical] + 0.5, color=VIEW_COLORS[PLANE_VIEWS[vertical]], lw=0.7, alpha=0.9)
        self._draw_annotations(ax, plane, index)
        if self.show_overlays.get():
            self._draw_overlay_text(ax, name, plane, index)
        if self.zoom_rect and self.zoom_rect["view"] == name:
            (x0, y0), (x1, y1) = self.zoom_rect["start"], self.zoom_rect["end"]
            ax.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="yellow", facecolor="none", linestyle="--")
            )
        view["canvas"].draw_idle()

    def _draw_scales(self, ax, name, plane, x_limits, y_limits):
        """Ticks in mm and anatomical axis labels, for a maximized view."""
        sx, sy = self.volume.pixel_spacing(plane)
        zoom = self.zoom[name]
        count = 12 if zoom > 10 else 10 if zoom > 5 else 8 if zoom > 2 else 6
        x_ticks, y_ticks = ticks(*x_limits, sx, count), ticks(*y_limits, sy, count)
        size = max(7, min(10, 60 // max(len(x_ticks), 1)))
        ax.set_xticks(x_ticks, [format_mm(x * sx) for x in x_ticks], fontsize=size, color="white")
        ax.set_yticks(y_ticks, [format_mm(y * sy) for y in y_ticks], fontsize=size, color="white")
        ax.tick_params(colors="white", length=4, width=0.5, pad=3)
        for spine in ax.spines.values():
            spine.set(color="gray", linewidth=0.5, visible=True, alpha=0.8)
        x_label, y_label = AXIS_LABELS[plane]
        ax.set_xlabel(f"{x_label} (mm)", fontsize=size + 1, color="lightgray", labelpad=5)
        ax.set_ylabel(f"{y_label} (mm)", fontsize=size + 1, color="lightgray", labelpad=5)

    def _draw_annotations(self, ax, plane, index):
        for kind, draw in ((DRAWING, draw_stroke), (MEASUREMENT, draw_measurement)):
            for item in self.store.on_slice(kind, plane, index):
                draw(ax, item, highlighted=self.highlight == (kind, item["id"]))
        area = self.pending_area
        if area and (area["plane"], area["slice"]) == (plane, index) and len(area["points"]) > 1:
            points = np.asarray(area["points"])
            ax.plot(points[:, 0], points[:, 1], color="yellow", linewidth=2, alpha=0.8, linestyle="--")
            draw_start_marker(ax, points[0])
        distance = self.pending_distance
        if distance and (distance["plane"], distance["slice"]) == (plane, index):
            draw_start_marker(ax, distance["p1"])

    def _draw_overlay_text(self, ax, name, plane, index):
        position_mm = index * self.volume.spacing[plane]
        text = (
            f"{name.upper()}\n"
            f"Slice: {index}/{self.volume.shape[plane] - 1} ({position_mm:.1f} mm)\n"
            f"W: {self.window:.0f} L: {self.level:.0f} | Z: {self.zoom[name]:.1f}x"
        )
        ax.text(
            0.02,
            0.98,
            text,
            color=VIEW_COLORS[name],
            fontsize=7,
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=TEXT_BOX,
        )
        ax.text(
            0.98,
            0.98,
            Path(self.volume.path).name,
            color="gray",
            fontsize=6,
            ha="right",
            va="top",
            transform=ax.transAxes,
            bbox=TEXT_BOX,
        )
        # Orientation letters at the edges, for example R and L for right and left.
        (x_start, _, x_end), (y_start, _, y_end) = (label.split() for label in AXIS_LABELS[plane])
        for x, y, letter in ((0.01, 0.5, x_start), (0.99, 0.5, x_end), (0.5, 0.01, y_start), (0.5, 0.99, y_end)):
            ax.text(
                x,
                y,
                letter[0],
                color="orange",
                fontsize=9,
                fontweight="bold",
                transform=ax.transAxes,
                ha="left" if x < 0.5 else "right" if x > 0.5 else "center",
                va="bottom" if y < 0.5 else "top" if y > 0.5 else "center",
            )

    def _save_snapshot(self, kind, item, path):
        """Save an image of one annotation on its slice, with its comment and value, for a report."""
        plane = item["plane"]
        image = self.volume.slice(plane, item["slice"])
        height, width = image.shape
        figure = Figure(figsize=(8, 8), dpi=150, facecolor="black")
        FigureCanvasAgg(figure)
        ax = figure.add_axes((0, 0, 1, 1), facecolor="black")
        ax.imshow(
            image,
            cmap=self.colormap.get(),
            vmin=self.vmin,
            vmax=self.vmax,
            origin="lower",
            extent=(0, width, 0, height),
        )
        ax.set_aspect(self.volume.aspect(plane))
        if kind == DRAWING:
            draw_stroke(ax, {**item, "size": item["size"] + 2, "color": "white", "opacity": 0.9})
            draw_stroke(ax, item)
            points, value = np.asarray(item["points"]), ""
        else:
            draw_measurement(ax, item, highlighted=True)
            if item.get("type") == "area":
                points, value = np.asarray(item["points"]), f"{item['area_mm2']:.2f} mm²"
            else:
                points, value = np.asarray([item["p1"], item["p2"]]), f"{item['dist_mm']:.2f} mm"
        center = points.mean(axis=0)
        half = max(np.ptp(points, axis=0).max(), 50) * 1.5
        ax.set_xlim(center[0] - half, center[0] + half)
        y_limits = (center[1] - half, center[1] + half)
        ax.set_ylim(*(y_limits[::-1] if plane in FLIPPED_VERTICAL else y_limits))
        _hide_axes(ax)
        caption = "\n".join(text for text in (item.get("comment", ""), value) if text)
        if caption:
            ax.text(
                0.5,
                0.02,
                caption,
                color="yellow",
                fontsize=10,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                bbox={**TEXT_BOX, "alpha": 0.7},
            )
        figure.savefig(path, facecolor="black", bbox_inches="tight", pad_inches=0.05)
