"""Session files: the annotations and the view settings of one volume, stored as JSON.

Version 2 sessions record the SHA-256 checksum of the reoriented volume (Volume.checksum).
Version 1 sessions (NeuroXAI 1.0 and 1.1) can still be read. Their checksum used Python's hash(),
which changes between runs, so it cannot be compared.
"""

import json
from datetime import datetime

import numpy as np

from .annotations import AnnotationStore

VERSION = "2.0"
REQUIRED_KEYS = ("session_info", "patient_info", "view_state", "drawings", "measurements")
VIEW_KEYS = ("current_slices", "window", "level")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that also writes NumPy numbers and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build(volume, store, patient, view_state, tool_state):
    """Session dictionary for a volume, its annotations and the view and tool settings."""
    drawings, measurements = store.to_session()
    return {
        "session_info": {
            "version": VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "volume_file": volume.path,
            "volume_checksum": volume.checksum(),
        },
        "patient_info": dict(patient),
        "view_state": view_state,
        "tool_state": tool_state,
        "drawings": drawings,
        "measurements": measurements,
    }


def write(path, session):
    with open(path, "w") as f:
        json.dump(session, f, indent=2, cls=NumpyEncoder)


def read(path):
    """Read and check a session file. Raise ValueError if it is not a NeuroXAI session."""
    with open(path) as f:
        session = json.load(f)
    missing = [key for key in REQUIRED_KEYS if key not in session] if isinstance(session, dict) else list(REQUIRED_KEYS)
    if not missing:
        missing = [f"view_state.{key}" for key in VIEW_KEYS if key not in session["view_state"]]
    if missing:
        raise ValueError(f"not a NeuroXAI session file (missing: {', '.join(missing)})")
    return session


def is_legacy(session):
    """True for a version 1 session, whose annotations used the orientation of the file."""
    return not str(session["session_info"].get("version", "1")).startswith("2")


def checksum_matches(session, volume):
    """False only if a version 2 session was saved for a different volume."""
    if is_legacy(session):
        return True
    return session["session_info"].get("volume_checksum") in (None, volume.checksum())


def annotations(session):
    return AnnotationStore.from_session(session["drawings"], session["measurements"])
