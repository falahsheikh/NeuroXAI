import pytest

from neuroxai.annotations import DRAWING, MEASUREMENT, AnnotationStore, summary
from neuroxai.volume import AXIAL, CORONAL

SPACING = (1.2, 1.5)


def ids(store, kind):
    return [item["id"] for item in store.items(kind)]


def test_measurements_use_the_pixel_spacing():
    store = AnnotationStore()
    distance = store.add_distance(AXIAL, 3, (0, 0), (3, 4), "red", SPACING)
    area = store.add_area(AXIAL, 3, [(0, 0), (10, 0), (10, 10), (0, 10)], "red", SPACING)
    assert distance["dist_mm"] == pytest.approx((3.6**2 + 6.0**2) ** 0.5)
    assert area["area_pixels"] == pytest.approx(100.0)
    assert area["area_mm2"] == pytest.approx(180.0)
    assert (distance["id"], area["id"]) == (1, 2)  # distances and areas share one id sequence
    assert summary(MEASUREMENT, area) == "(Axial 3) Area 180.00 mm²"


def test_strokes_record_points_that_are_far_enough_apart():
    store = AnnotationStore()
    stroke = store.begin_stroke(CORONAL, 7, (0, 0), "yellow", 2.0, 0.8)
    assert store.extend_stroke(stroke, (0.5, 0.5)) is False
    assert store.extend_stroke(stroke, (2, 0)) is True
    assert stroke["points"] == [(0, 0), (2, 0)]
    assert store.on_slice(DRAWING, CORONAL, 7) == [stroke]
    assert store.on_slice(DRAWING, CORONAL, 8) == []


def test_undo_and_redo_of_each_action():
    store = AnnotationStore()
    first = store.add_distance(AXIAL, 0, (0, 0), (1, 0), "red", SPACING)
    second = store.add_distance(AXIAL, 0, (0, 0), (2, 0), "red", SPACING)
    store.set_comment(MEASUREMENT, first["id"], "note")
    store.delete(MEASUREMENT, first["id"])
    assert ids(store, MEASUREMENT) == [second["id"]]

    assert store.undo() == "deletion of distance 1"
    assert ids(store, MEASUREMENT) == [1, 2]  # restored at its old position
    assert store.undo() == "comment on distance 1"
    assert "comment" not in first
    assert store.redo() == "comment on distance 1"
    assert first["comment"] == "note"
    assert store.undo() and store.undo() and store.undo()
    assert store.items(MEASUREMENT) == []
    assert store.undo() is None
    assert store.redo() == "distance 1"
    assert ids(store, MEASUREMENT) == [1]


def test_clear_undo_redo_undo_restores_the_items():
    # NeuroXAI 1.1 lost the items when a clear was undone, redone and undone again.
    store = AnnotationStore()
    for i in range(3):
        store.begin_stroke(AXIAL, i, (0, 0), "yellow", 2.0, 0.8)
    store.clear(DRAWING)
    for _ in range(2):
        store.undo()
        assert ids(store, DRAWING) == [1, 2, 3]
        store.redo()
        assert store.items(DRAWING) == []
    store.undo()
    assert ids(store, DRAWING) == [1, 2, 3]


def test_a_new_action_clears_the_redo_history():
    store = AnnotationStore()
    store.begin_stroke(AXIAL, 0, (0, 0), "yellow", 2.0, 0.8)
    store.undo()
    assert store.can_redo
    store.begin_stroke(AXIAL, 0, (1, 1), "yellow", 2.0, 0.8)
    assert not store.can_redo
    assert ids(store, DRAWING) == [2]  # ids are not reused


def test_comments():
    store = AnnotationStore()
    item = store.add_distance(AXIAL, 0, (0, 0), (1, 0), "red", SPACING)
    assert store.set_comment(MEASUREMENT, item["id"], "  first ") is item
    assert item["comment"] == "first"
    assert store.set_comment(MEASUREMENT, item["id"], "first") is None  # no change, no action
    store.set_comment(MEASUREMENT, item["id"], "")
    assert "comment" not in item
    assert store.set_comment(MEASUREMENT, 99, "x") is None
    assert store.latest(MEASUREMENT) is item
    assert store.latest(DRAWING) is None


def test_session_round_trip_keeps_ids_and_values():
    store = AnnotationStore()
    store.begin_stroke(AXIAL, 1, (0, 0), "yellow", 2.0, 0.8)
    store.begin_stroke(CORONAL, 2, (5, 5), "red", 3.0, 0.5)
    store.add_area(AXIAL, 1, [(0, 0), (4, 0), (0, 3)], "lime", SPACING)
    drawings, measurements = store.to_session()
    assert set(drawings) == {str(AXIAL), str(CORONAL)}

    loaded = AnnotationStore.from_session(drawings, measurements)
    assert loaded.strokes == store.strokes
    assert loaded.measurements == store.measurements
    assert not loaded.can_undo
    assert loaded.begin_stroke(AXIAL, 0, (0, 0), "yellow", 2.0, 0.8)["id"] == 3
    assert loaded.add_distance(AXIAL, 0, (0, 0), (1, 1), "red", SPACING)["id"] == 2
