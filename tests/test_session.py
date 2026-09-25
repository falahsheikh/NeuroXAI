import json

import numpy as np
import pytest

from neuroxai import session
from neuroxai.annotations import MEASUREMENT, AnnotationStore
from neuroxai.volume import AXIAL, Volume

VIEW_STATE = {"current_slices": [1, 2, 3], "window": 100.0, "level": 50.0}


def make_volume(path="volume.nii.gz", value=0.0):
    return Volume(np.full((4, 5, 6), value, dtype=np.float32), (1.0, 1.0, 1.0), path)


def test_round_trip(tmp_path):
    volume, store = make_volume(), AnnotationStore()
    item = store.add_distance(AXIAL, 1, (0, 0), (np.float64(3.0), 4), "red", (1.0, 1.0))
    store.set_comment(MEASUREMENT, item["id"], "note")
    path = tmp_path / "session.json"
    session.write(path, session.build(volume, store, {"name": "A", "id": "1"}, VIEW_STATE, {"draw_color": "red"}))

    data = session.read(path)
    assert data["session_info"]["version"] == session.VERSION
    assert not session.is_legacy(data)
    assert session.checksum_matches(data, volume)
    assert not session.checksum_matches(data, make_volume(value=1.0))
    loaded = session.annotations(data)
    assert loaded.measurements == store.measurements


def test_version_1_sessions_are_read_without_the_checksum(tmp_path):
    legacy = {
        "session_info": {"version": "1.0", "volume_file": "x.nii", "volume_checksum": "-123456789"},
        "patient_info": {"name": "N/A", "id": "N/A"},
        "view_state": VIEW_STATE,
        "drawings": {
            "0": {"1": [{"id": 4, "plane": 0, "slice": 1, "points": [[0, 0], [1, 1]], "color": "red", "size": 2}]}
        },
        "measurements": [{"id": 2, "plane": 0, "slice": 1, "color": "red", "p1": [0, 0], "p2": [1, 0], "dist_mm": 1.0}],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy))
    data = session.read(path)
    assert session.is_legacy(data)
    assert session.checksum_matches(data, make_volume())
    store = session.annotations(data)
    assert store.measurements[0]["type"] == "distance"
    assert store.measurements[0]["p1"] == (0, 0)
    assert store.strokes[0]["points"] == [(0, 0), (1, 1)]


@pytest.mark.parametrize("content", ["[]", '{"session_info": {}}', "not json"])
def test_invalid_files_raise_value_error(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        session.read(path)
