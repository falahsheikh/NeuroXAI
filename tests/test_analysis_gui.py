"""The Analysis Tool with the bundled model. The dialogs are replaced by functions."""

import json
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pytest
from PIL import Image
from tensorflow import keras

from neuroxai.config import DEFAULT_MODEL


@pytest.fixture
def app(tk_root):
    from neuroxai.analysis.app import AnalysisApp

    tool = AnalysisApp(tk_root)
    assert tool.load_model(DEFAULT_MODEL, interactive=False)
    tk_root.update()
    return tool


@pytest.fixture
def slice_file(tmp_path):
    rows, cols = np.indices((256, 256))
    brain = ((rows - 128) / 100.0) ** 2 + ((cols - 128) / 80.0) ** 2 < 1
    path = tmp_path / "slice.png"
    Image.fromarray((brain * np.random.default_rng(0).uniform(40, 200, size=brain.shape)).astype(np.uint8)).save(path)
    return path


def test_analysis_panes_and_report(app, slice_file, tmp_path, monkeypatch):
    app.analyze_file(slice_file)
    assert app.analysis is not None
    assert all(pane.figure is not None for pane in app.panes.values())
    prediction = app.panes["Prediction"]
    assert prediction.shows_placeholder

    app.toggle_maximize(prediction)
    assert app.maximized_pane is prediction and not prediction.shows_placeholder
    app.toggle_maximize(app.panes["Summary"])
    assert prediction.shows_placeholder and app.maximized_pane is app.panes["Summary"]
    app.reset_view()
    assert app.maximized_pane is None

    monkeypatch.setattr(filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(messagebox, "showinfo", lambda *args, **kwargs: None)
    app.save_report()
    (report_dir,) = tmp_path.glob("neuroxai_report_*")
    assert len(list((report_dir / "figures").glob("*.png"))) == len(app.panes)
    assert json.loads((report_dir / "summary.json").read_text())["predicted_class"] == app.analysis.label

    app.clear()
    assert app.analysis is None and all(pane.figure is None for pane in app.panes.values())


def test_layer_dialog_changes_the_layers(app, slice_file, tk_root):
    from neuroxai.analysis.layer_dialog import LayerDialog

    app.analyze_file(slice_file)
    dialog = LayerDialog(tk_root, app.explainer)
    dialog.window = tk.Toplevel(tk_root)
    dialog._build()
    dialog._set_all("top_conv")
    dialog._apply(reanalyze=True)
    assert dialog.result == {"layers": {"gradcam": "top_conv", "guided_gradcam": "top_conv"}, "reanalyze": True}
    app._analyze(app.analysis.image, app.analysis.image_path)
    assert app.analysis.layers["gradcam"] == "top_conv"


def test_unusable_models_are_rejected(app, tmp_path, monkeypatch):
    inputs = keras.Input((32, 32, 3))
    pooled = keras.layers.GlobalAveragePooling2D()(keras.layers.Conv2D(2, 3)(inputs))
    path = tmp_path / "two_classes.keras"
    keras.Model(inputs, keras.layers.Dense(2, activation="softmax")(pooled)).save(path)
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, message, **kwargs: errors.append(message))
    assert not app.load_model(path)
    assert "2 outputs" in errors[0]
    assert app.model_path == DEFAULT_MODEL  # the model that was loaded stays
