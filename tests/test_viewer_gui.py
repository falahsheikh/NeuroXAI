"""The Slice Viewer, driven with synthetic mouse events. The dialogs are replaced by functions."""

import json
import math
from tkinter import filedialog, messagebox, simpledialog
from types import SimpleNamespace

import pytest
from conftest import SIZE, SPACING

from neuroxai.annotations import DRAWING, MEASUREMENT
from neuroxai.volume import AXIAL, CORONAL, SAGITTAL


@pytest.fixture
def viewer(tk_root, ras_volume_file):
    from neuroxai.viewer.app import SliceViewer

    app = SliceViewer(tk_root)
    assert app.open_volume(ras_volume_file)
    tk_root.update()
    return app


def mouse(viewer, view, x, y, button=1, key=None):
    """A matplotlib mouse event at data coordinates (x, y) of a view."""
    ax = viewer.views[view]["ax"]
    px, py = ax.transData.transform((x, y))
    return SimpleNamespace(inaxes=ax, xdata=x, ydata=y, x=px, y=py, button=button, key=key)


def drag(viewer, view, points, button=1):
    viewer._on_press(mouse(viewer, view, *points[0], button), view)
    for point in points[1:]:
        viewer._on_motion(mouse(viewer, view, *point, button), view)
    viewer._on_release(mouse(viewer, view, *points[-1], button), view)


def test_volume_views_and_crosshair(viewer):
    nx, ny, nz = SIZE
    sx, sy, sz = SPACING
    assert viewer.volume.shape == (nz, ny, nx)
    assert viewer.current_slices == [nz // 2, ny // 2, nx // 2]
    axial = viewer.views["axial"]["ax"]
    assert axial.get_aspect() == pytest.approx(sy / sx)
    assert axial.get_ylim()[0] > axial.get_ylim()[1]  # anterior at the top
    assert viewer.views["sagittal"]["ax"].get_aspect() == pytest.approx(sz / sy)

    viewer._on_press(mouse(viewer, "axial", 5.7, 8.2), "axial")
    viewer._on_release(mouse(viewer, "axial", 5.7, 8.2), "axial")
    assert (viewer.current_slices[SAGITTAL], viewer.current_slices[CORONAL]) == (5, 8)
    viewer._on_motion(mouse(viewer, "coronal", 3.5, 7.5), "coronal")  # x = 3 and z = 7 in the coronal view
    y = viewer.current_slices[CORONAL]
    assert viewer.position_label.cget("text") == f"Voxel (x, y, z): 3, {y}, 7"
    assert viewer.intensity_label.cget("text") == f"Intensity: {viewer.volume.data[7, y, 3]:.2f}"

    viewer._on_scroll(mouse(viewer, "axial", 5, 5, button="up"), "axial")
    assert viewer.current_slices[AXIAL] == nz // 2 - 1
    viewer._on_scroll(mouse(viewer, "axial", 5, 5, button="up", key="control"), "axial")
    assert viewer.zoom["axial"] == pytest.approx(1.2)


def test_tool_modes_are_exclusive(viewer):
    viewer.modes["draw"].set(True)
    viewer.modes["measure"].set(True)
    assert viewer.active_mode() == "measure"
    viewer.modes["measure"].set(False)
    assert viewer.active_mode() is None


def test_annotations_comments_undo_and_redo(viewer, monkeypatch):
    sx, sy, _ = SPACING
    viewer.modes["draw"].set(True)
    drag(viewer, "axial", [(2, 2), (5, 5), (9, 9)])
    assert viewer.store.strokes[0]["points"] == [(2, 2), (5, 5), (9, 9)]

    viewer.modes["measure"].set(True)
    for point in ((0, 0), (3, 4)):
        viewer._on_press(mouse(viewer, "axial", *point), "axial")
    distance = viewer.store.latest(MEASUREMENT)
    assert distance["dist_mm"] == pytest.approx(math.hypot(3 * sx, 4 * sy))

    viewer.modes["area"].set(True)
    drag(viewer, "axial", [(0, 0), (10, 0), (10, 10), (0, 10)])
    area = viewer.store.latest(MEASUREMENT)
    assert area["area_mm2"] == pytest.approx(100 * sx * sy)
    assert len(viewer.trees[MEASUREMENT].get_children()) == 2

    monkeypatch.setattr(simpledialog, "askstring", lambda *args, **kwargs: "hippocampus")
    viewer.comment_annotation(MEASUREMENT)  # nothing is selected, so the newest item gets the comment
    assert area["comment"] == "hippocampus"
    assert "hippocampus" in viewer.trees[MEASUREMENT].item(f"Measure_{area['id']}", "values")[0]

    monkeypatch.setattr(messagebox, "askyesno", lambda *args, **kwargs: True)
    viewer.trees[MEASUREMENT].selection_set(f"Measure_{distance['id']}")
    viewer.delete_annotation(MEASUREMENT)
    assert viewer.store.measurements == [area]
    viewer.undo()
    assert viewer.store.measurements == [distance, area]
    viewer.redo()
    assert viewer.store.measurements == [area]

    viewer.clear_annotations(DRAWING, confirm=False)
    viewer.undo()
    viewer.redo()
    viewer.undo()
    assert len(viewer.store.strokes) == 1
    assert str(viewer.undo_button.cget("state")) == "normal"


def test_escape_cancels_a_measurement(viewer):
    viewer.modes["measure"].set(True)
    viewer._on_press(mouse(viewer, "axial", 1, 1), "axial")
    assert viewer.pending_distance is not None
    viewer.cancel_pending()
    assert viewer.pending_distance is None
    assert viewer.store.measurements == []


def test_zoom_pan_and_window_level(viewer):
    viewer.modes["zoom_select"].set(True)
    drag(viewer, "coronal", [(2, 2), (8, 12)])
    assert viewer.zoom["coronal"] > 1
    viewer.modes["pan"].set(True)
    center = viewer.zoom_center["coronal"]
    drag(viewer, "coronal", [(5, 5), (6, 5)])
    assert viewer.zoom_center["coronal"][0] < center[0]
    viewer.reset_zoom()
    assert viewer.zoom["coronal"] == 1.0

    viewer.modes["pan"].set(False)
    window = viewer.window
    viewer._on_press(mouse(viewer, "axial", 5, 5, button=3), "axial")
    event = mouse(viewer, "axial", 5, 5, button=3)
    event.x += 20
    viewer._on_motion(event, "axial")
    assert viewer.window > window


def test_maximize_keeps_the_order_of_the_views(viewer):
    def order(side):
        return [str(pane) for pane in viewer.column_panes[side].panes()]

    before = {side: order(side) for side in ("left", "right")}
    for name in ("sagittal", "axial", "coronal"):
        viewer.toggle_maximize(name)
        assert viewer.maximized_view == name
        viewer.toggle_maximize(name)
        assert {side: order(side) for side in ("left", "right")} == before


def test_session_and_report(viewer, tk_root, tmp_path, monkeypatch):
    from neuroxai.viewer.app import SliceViewer

    viewer.modes["area"].set(True)
    drag(viewer, "coronal", [(1, 1), (6, 1), (6, 6)])
    viewer.store.set_comment(MEASUREMENT, 1, "lesion")
    viewer.current_slices[SAGITTAL] = 4
    session_file = tmp_path / "session.json"
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kwargs: str(session_file))
    viewer.save_session()

    other = SliceViewer(tk_root)
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kwargs: str(session_file))
    other.load_session()
    assert other.store.measurements == viewer.store.measurements
    assert other.current_slices[SAGITTAL] == 4

    # A report with an area measurement failed in NeuroXAI 1.1.
    monkeypatch.setattr(filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(messagebox, "showinfo", lambda *args, **kwargs: None)
    other.save_report()
    (report_dir,) = tmp_path.glob("Report_*")
    report = json.loads((report_dir / "report.json").read_text())
    assert report["measurements"][0]["comment"] == "lesion"
    assert (report_dir / report["measurements"][0]["image_file"]).stat().st_size > 0
