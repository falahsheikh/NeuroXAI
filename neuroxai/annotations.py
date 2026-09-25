"""Drawings and measurements on slices, with undo and redo.

The annotations are dictionaries, so that sessions and reports can store them as JSON:

- drawing (stroke): id, plane, slice, points, color, size, opacity
- distance measurement: id, type="distance", plane, slice, p1, p2, color, dist_mm
- area measurement: id, type="area", plane, slice, points, color, area_pixels, area_mm2

Each annotation can also have a "comment". Points are (x, y) slice pixel coordinates.
Drawings and measurements have separate id sequences.
"""

import math

from .geometry import area_mm2, distance_mm, polygon_area
from .volume import PLANE_NAMES

DRAWING, MEASUREMENT = "Drawing", "Measure"
KINDS = (DRAWING, MEASUREMENT)


def describe(kind, item):
    """Short name of an annotation, for example "distance 3"."""
    if kind == DRAWING:
        return f"drawing {item['id']}"
    return f"{item.get('type', 'distance')} {item['id']}"


def summary(kind, item):
    """One line that identifies an annotation and its value, for lists and reports."""
    text = f"({PLANE_NAMES[item['plane']]} {item['slice']}) "
    if kind == DRAWING:
        text += f"Drawing {item['id']}"
    elif item.get("type") == "area":
        text += f"Area {item['area_mm2']:.2f} mm²"
    else:
        text += f"Distance {item['dist_mm']:.2f} mm"
    return f"{text} - {item['comment']}" if item.get("comment") else text


class AnnotationStore:
    """The annotations of one volume. Each change is one action that undo() and redo() can reverse."""

    def __init__(self):
        self.strokes = []
        self.measurements = []
        self._next_id = dict.fromkeys(KINDS, 1)
        self._undo = []  # (description, do, undo) for each action, oldest first
        self._redo = []

    # Queries

    def items(self, kind):
        return self.strokes if kind == DRAWING else self.measurements

    def find(self, kind, item_id):
        return next((item for item in self.items(kind) if item["id"] == item_id), None)

    def on_slice(self, kind, plane, slice_index):
        return [item for item in self.items(kind) if item["plane"] == plane and item["slice"] == slice_index]

    def latest(self, kind):
        """The annotation of this kind with the highest id, that is, the most recent one."""
        return max(self.items(kind), key=lambda item: item["id"], default=None)

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    # Changes. Each returns the new or changed annotation, or a falsy value if nothing changed.

    def begin_stroke(self, plane, slice_index, point, color, size, opacity):
        """Start a drawing. Add points with extend_stroke() while the mouse button is down."""
        stroke = {
            "id": self._take_id(DRAWING),
            "plane": plane,
            "slice": slice_index,
            "points": [tuple(point)],
            "color": color,
            "size": size,
            "opacity": opacity,
        }
        self._add(DRAWING, stroke)
        return stroke

    @staticmethod
    def extend_stroke(stroke, point, min_step=1.0):
        """Add point to a drawing if it is more than min_step pixels from the last point."""
        last = stroke["points"][-1]
        if math.dist(last, point) <= min_step:
            return False
        stroke["points"].append(tuple(point))
        return True

    def add_distance(self, plane, slice_index, p1, p2, color, pixel_spacing):
        measurement = {
            "id": self._take_id(MEASUREMENT),
            "type": "distance",
            "plane": plane,
            "slice": slice_index,
            "p1": tuple(p1),
            "p2": tuple(p2),
            "color": color,
            "dist_mm": distance_mm(p1, p2, pixel_spacing),
        }
        self._add(MEASUREMENT, measurement)
        return measurement

    def add_area(self, plane, slice_index, points, color, pixel_spacing):
        """Add the area inside a closed outline. The last point connects to the first point."""
        points = [tuple(p) for p in points]
        measurement = {
            "id": self._take_id(MEASUREMENT),
            "type": "area",
            "plane": plane,
            "slice": slice_index,
            "points": points,
            "color": color,
            "area_pixels": polygon_area(points),
            "area_mm2": area_mm2(points, pixel_spacing),
        }
        self._add(MEASUREMENT, measurement)
        return measurement

    def delete(self, kind, item_id):
        items, item = self.items(kind), self.find(kind, item_id)
        if item is None:
            return None
        index = items.index(item)
        self._do(f"deletion of {describe(kind, item)}", lambda: items.remove(item), lambda: items.insert(index, item))
        return item

    def set_comment(self, kind, item_id, comment):
        """Set or change the comment of an annotation. An empty comment removes the comment."""
        item = self.find(kind, item_id)
        new, old = (comment or "").strip() or None, item.get("comment") if item else None
        if item is None or new == old:
            return None

        def set_to(value):
            if value is None:
                item.pop("comment", None)
            else:
                item["comment"] = value

        self._do(f"comment on {describe(kind, item)}", lambda: set_to(new), lambda: set_to(old))
        return item

    def clear(self, kind):
        items = self.items(kind)
        saved = list(items)
        if not saved:
            return False
        label = "drawings" if kind == DRAWING else "measurements"
        self._do(f"clearing of all {label}", items.clear, lambda: items.extend(saved))
        return True

    def undo(self):
        """Reverse the last action. Return its description, or None if there is nothing to undo."""
        if not self._undo:
            return None
        action = self._undo.pop()
        action[2]()
        self._redo.append(action)
        return action[0]

    def redo(self):
        """Do the last undone action again. Return its description, or None if there is nothing to redo."""
        if not self._redo:
            return None
        action = self._redo.pop()
        action[1]()
        self._undo.append(action)
        return action[0]

    # Sessions

    def to_session(self):
        """(drawings, measurements) in the session file format: drawings are grouped by plane and slice."""
        drawings = {}
        for stroke in self.strokes:
            drawings.setdefault(str(stroke["plane"]), {}).setdefault(str(stroke["slice"]), []).append(stroke)
        return drawings, list(self.measurements)

    @classmethod
    def from_session(cls, drawings, measurements):
        """A store with the annotations of a session file and an empty history."""
        store = cls()
        for slices in drawings.values():
            for strokes in slices.values():
                for stroke in strokes:
                    store.strokes.append({**stroke, "points": [tuple(p) for p in stroke["points"]]})
        for measurement in measurements:
            item = {**measurement, "type": measurement.get("type", "distance")}
            for key in ("p1", "p2"):
                if key in item:
                    item[key] = tuple(item[key])
            if "points" in item:
                item["points"] = [tuple(p) for p in item["points"]]
            store.measurements.append(item)
        store.strokes.sort(key=lambda s: s["id"])
        for kind in KINDS:
            store._next_id[kind] = max((item["id"] for item in store.items(kind)), default=0) + 1
        return store

    # Internal

    def _take_id(self, kind):
        item_id = self._next_id[kind]
        self._next_id[kind] += 1
        return item_id

    def _add(self, kind, item):
        items = self.items(kind)
        self._do(describe(kind, item), lambda: items.append(item), lambda: items.remove(item))

    def _do(self, description, do, undo):
        do()
        self._undo.append((description, do, undo))
        self._redo.clear()
