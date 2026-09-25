"""Analysis of one slice: the prediction and the explanation maps. This module does not use Tkinter."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import xai
from ..config import CLASS_DESCRIPTIONS, CLASS_NAMES, RESEARCH_NOTICE

MAPS = {"gradcam": "Grad-CAM++", "guided_gradcam": "Guided Grad-CAM++", "consensus": "Consensus"}


@dataclass
class Analysis:
    """The result of one analysis. The maps are in [0, 1] and have the size of the image."""

    image: np.ndarray  # RGB uint8 model input
    probabilities: np.ndarray
    maps: dict  # name in MAPS -> map
    class_maps: list  # Grad-CAM++ map for each class, at the Grad-CAM++ layer
    mask: np.ndarray  # 1 inside the brain
    layers: dict  # method in xai.METHODS -> layer name
    image_path: str = ""
    model_name: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat(sep=" ", timespec="seconds"))

    @property
    def predicted(self):
        return int(np.argmax(self.probabilities))

    @property
    def label(self):
        return CLASS_NAMES[self.predicted]

    @property
    def confidence(self):
        return float(self.probabilities[self.predicted])

    def masked(self, name):
        """The map with the values outside the brain set to 0."""
        return self.maps[name] * self.mask

    def overlay(self, name):
        return xai.overlay(xai.normalize(self.masked(name)), self.image)

    def peak_location(self, name="gradcam"):
        """Position of the maximum of the masked map in the image, in words (for example "upper left")."""
        height, width = self.mask.shape
        row, col = np.unravel_index(np.argmax(self.masked(name)), (height, width))
        vertical = ("upper", "middle", "lower")[min(3 * row // height, 2)]
        horizontal = ("left", "center", "right")[min(3 * col // width, 2)]
        return "center" if (vertical, horizontal) == ("middle", "center") else f"{vertical} {horizontal}"

    def spread(self, name="gradcam"):
        """Fraction of the brain pixels where the masked map is 0.1 or more."""
        inside = self.mask > 0
        return float((self.masked(name)[inside] >= 0.1).mean()) if inside.any() else 0.0

    def to_dict(self):
        """The results without the maps, for the saved report."""
        return {
            "image": self.image_path,
            "model": self.model_name,
            "analyzed": self.created,
            "predicted_class": self.label,
            "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, self.probabilities, strict=True)},
            "layers": {xai.METHODS[m]: layer for m, layer in self.layers.items()},
            "gradcam_peak": self.peak_location(),
            "gradcam_spread": round(self.spread(), 4),
            "notice": RESEARCH_NOTICE,
        }


def analyze(model, explainer, preprocess, image, image_path="", model_name=""):
    """Classify one RGB uint8 slice and compute its explanation maps for the predicted class."""
    batch = preprocess(image.astype(np.float32)[np.newaxis])
    probabilities = model.predict(batch, verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    gradcam = explainer.gradcam(batch, predicted)
    guided = explainer.guided_gradcam(batch, predicted)
    class_maps = [gradcam if c == predicted else explainer.gradcam(batch, c) for c in range(len(probabilities))]
    return Analysis(
        image=image,
        probabilities=probabilities,
        maps={"gradcam": gradcam, "guided_gradcam": guided, "consensus": xai.consensus([gradcam, guided])},
        class_maps=class_maps,
        mask=xai.brain_mask(image),
        layers=dict(explainer.selected),
        image_path=str(image_path),
        model_name=model_name,
    )


def summary_text(analysis):
    """Plain-text summary of an analysis, for the Summary pane and the saved report."""
    lines = [
        "NEUROXAI ANALYSIS SUMMARY",
        RESEARCH_NOTICE,
        "",
        f"Image:     {Path(analysis.image_path).name or '-'}",
        f"Model:     {analysis.model_name or '-'}",
        f"Analyzed:  {analysis.created}",
        "",
        f"Predicted class: {analysis.label} ({CLASS_DESCRIPTIONS[analysis.label]})",
        f"Probability:     {analysis.confidence:.3f}",
        "",
        "Class probabilities:",
    ]
    for i, (name, p) in enumerate(zip(CLASS_NAMES, analysis.probabilities, strict=True)):
        lines.append(f"{'*' if i == analysis.predicted else ' '} {name:<5} {p:.3f}")
    lines += ["", "Layers:"]
    lines += [f"  {xai.METHODS[m]:<18} {layer}" for m, layer in analysis.layers.items()]
    lines += [
        "",
        f"Grad-CAM++ peak:    {analysis.peak_location()} of the image",
        f"Grad-CAM++ spread:  {analysis.spread():.0%} of the brain at 0.1 or more",
    ]
    return "\n".join(lines)
