"""The bundled model of the paper, on a synthetic slice."""

import numpy as np
import pytest
from tensorflow import keras

from neuroxai import xai
from neuroxai.analysis import charts
from neuroxai.analysis.pipeline import analyze, summary_text
from neuroxai.config import CLASS_NAMES, DEFAULT_MODEL, RESEARCH_NOTICE
from neuroxai.imaging import crop_and_resize_slice, to_uint8
from neuroxai.model_check import check_model


@pytest.fixture(scope="module")
def model():
    return keras.models.load_model(DEFAULT_MODEL)


def synthetic_slice():
    rows, cols = np.indices((256, 256))
    brain = ((rows - 128) / 100.0) ** 2 + ((cols - 128) / 80.0) ** 2 < 1
    gray = to_uint8(crop_and_resize_slice(brain * np.random.default_rng(0).uniform(40, 200, size=brain.shape)))
    return np.repeat(gray[..., None], 3, axis=-1)


def test_bundled_model(model):
    info = check_model(model)
    assert info.parameters == 6_083_667
    assert info.backbone == "EfficientNetV2B0"
    assert (info.errors, info.warnings) == ([], [])
    assert info.default_layer == "top_activation"
    assert "block6h_se_expand" not in info.layers  # 1x1 squeeze-and-excitation layers cannot be used
    assert info.layers[-2:] == ["top_conv", "top_activation"]


def test_analysis(model):
    info = check_model(model)
    analysis = analyze(model, xai.Explainer(model), info.preprocess, synthetic_slice(), "slice.png", "model.keras")
    np.testing.assert_allclose(analysis.probabilities.sum(), 1.0, rtol=1e-5)
    assert analysis.label in CLASS_NAMES
    assert len(analysis.class_maps) == len(CLASS_NAMES)
    for name, heatmap in analysis.maps.items():
        assert heatmap.shape == (224, 224), name
        assert np.isfinite(heatmap).all() and heatmap.min() >= 0 and heatmap.max() <= 1 + 1e-6
    assert analysis.masked("gradcam")[0, 0] == 0  # outside the brain
    assert RESEARCH_NOTICE in summary_text(analysis)
    assert analysis.to_dict()["layers"] == {"Grad-CAM++": "top_activation", "Guided Grad-CAM++": "top_activation"}

    figures = charts.build_all(analysis)
    assert list(figures) == [title for title, _ in charts.PANES]
