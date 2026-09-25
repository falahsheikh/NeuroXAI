"""Matplotlib figures for the panes of the Analysis Tool. This module does not use Tkinter."""

import textwrap
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .. import xai
from ..config import CLASS_DESCRIPTIONS, CLASS_NAMES, RESEARCH_NOTICE
from .pipeline import MAPS, summary_text

# (title, True if the pane shows a placeholder until it is maximized), row by row.
# Charts with much text or small parts show a placeholder, because a small pane cannot show them clearly.
PANES = (
    ("Original Slice", False),
    ("Prediction", True),
    ("Class Maps", False),
    ("Class Probabilities", True),
    ("Grad-CAM++ Raw", False),
    ("Grad-CAM++ Masked", False),
    ("Grad-CAM++ Overlay", False),
    ("Grad-CAM++ Stats", True),
    ("Guided Grad-CAM++ Raw", False),
    ("Guided Grad-CAM++ Masked", False),
    ("Guided Grad-CAM++ Overlay", False),
    ("Guided Grad-CAM++ Stats", True),
    ("Consensus Raw", False),
    ("Consensus Masked", False),
    ("Consensus Overlay", False),
    ("Summary", True),
)
COLUMNS = 4

TEXT_BOX = {"boxstyle": "round,pad=0.5", "facecolor": "#f8f9fa", "alpha": 0.8}
PREDICTED, OTHER = "#e74c3c", "#95a5a6"


def _shorten(text, width=60):
    """Text with its middle replaced by "..." if it is longer than width. File names have no spaces to wrap at."""
    if len(text) <= width:
        return text
    half = (width - 3) // 2
    return f"{text[:half]}...{text[-half:]}"


def _figure(width, height):
    return Figure(figsize=(width, height), layout="tight")


def _text_panel(ax, text, fontsize=9):
    ax.axis("off")
    ax.text(
        0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize, va="top", fontfamily="monospace", bbox=TEXT_BOX
    )


def original_slice(analysis):
    figure = _figure(6, 6)
    ax = figure.add_subplot()
    ax.imshow(analysis.image)
    ax.set_title(_shorten(Path(analysis.image_path).name or "Slice"), fontsize=8)
    ax.set_xlabel("Pixels", fontsize=10)
    ax.set_ylabel("Pixels", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    return figure


def prediction(analysis):
    figure = _figure(10, 5)
    bars_ax, text_ax = figure.subplots(1, 2)
    colors = [PREDICTED if i == analysis.predicted else OTHER for i in range(len(CLASS_NAMES))]
    bars = bars_ax.barh(CLASS_NAMES, analysis.probabilities, color=colors)
    for bar, p in zip(bars, analysis.probabilities, strict=True):
        inside = p > 0.75  # write the value inside a long bar, so that it stays in the axes
        bars_ax.text(
            p - 0.01 if inside else p + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.3f}",
            ha="right" if inside else "left",
            va="center",
            color="white" if inside else "black",
            fontsize=9,
        )
    bars_ax.set_xlim(0, 1)
    bars_ax.set_xlabel("Probability", fontsize=10)
    bars_ax.set_title("Class Probabilities", fontsize=12, fontweight="bold")
    bars_ax.grid(True, alpha=0.3)
    text = (
        "PREDICTION\n\n"
        f"Class:       {analysis.label}\n"
        f"             {CLASS_DESCRIPTIONS[analysis.label]}\n"
        f"Probability: {analysis.confidence:.1%}\n\n" + textwrap.fill(RESEARCH_NOTICE, 40)
    )
    _text_panel(text_ax, text, fontsize=10)
    return figure


def class_maps(analysis):
    """Grad-CAM++ overlay for each class, so that the evidence for the classes can be compared."""
    figure = _figure(9, 3.6)
    axes = figure.subplots(1, len(CLASS_NAMES))
    for i, (ax, name, heatmap) in enumerate(zip(axes, CLASS_NAMES, analysis.class_maps, strict=True)):
        ax.imshow(xai.overlay(xai.normalize(heatmap * analysis.mask), analysis.image))
        weight = "bold" if i == analysis.predicted else "normal"
        ax.set_title(f"{name}  p = {analysis.probabilities[i]:.3f}", fontsize=10, fontweight=weight)
        ax.axis("off")
    figure.suptitle("Grad-CAM++ for each class", fontsize=10)
    return figure


def class_probabilities(analysis):
    figure = _figure(10, 8)
    (pie_ax, bar_ax), (certainty_ax, text_ax) = figure.subplots(2, 2)
    p = np.asarray(analysis.probabilities, dtype=np.float64)
    colors = [f"C{i}" for i in range(len(p))]

    pie_ax.pie(
        p, labels=CLASS_NAMES, autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "", colors=colors, startangle=90
    )
    pie_ax.set_title("Class Distribution", fontsize=12, fontweight="bold")

    bars = bar_ax.bar(CLASS_NAMES, p, color=colors, alpha=0.7, edgecolor="black")
    for bar, value in zip(bars, p, strict=True):
        bar_ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.01, 0.95), f"{value:.4f}", ha="center", fontsize=9)
    bar_ax.set_ylim(0, 1)
    bar_ax.set_ylabel("Probability", fontsize=10)
    bar_ax.set_title("Probability Scores", fontsize=12, fontweight="bold")
    bar_ax.grid(True, alpha=0.3)

    # Normalized entropy: 0 when one class has all the probability, 1 when the classes are equally probable.
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    uncertainty = entropy / np.log(len(p))
    certainty_ax.bar(
        ["Certainty", "Uncertainty"], [1 - uncertainty, uncertainty], color=["#2ecc71", "#e74c3c"], alpha=0.7
    )
    certainty_ax.set_ylim(0, 1)
    certainty_ax.set_ylabel("Score", fontsize=10)
    certainty_ax.set_title("Certainty (1 - normalized entropy)", fontsize=12, fontweight="bold")
    certainty_ax.grid(True, alpha=0.3)

    text = (
        "STATISTICS\n\n"
        f"Max probability:    {p.max():.4f}\n"
        f"Min probability:    {p.min():.4f}\n"
        f"Mean probability:   {p.mean():.4f}\n"
        f"Std. deviation:     {p.std():.4f}\n\n"
        f"Entropy:            {entropy:.4f}\n"
        f"Normalized entropy: {uncertainty:.4f}"
    )
    _text_panel(text_ax, text)
    return figure


def heatmap(values, label):
    figure = _figure(4, 4)
    ax = figure.add_subplot()
    image = ax.imshow(values, cmap="jet", vmin=0, vmax=1)
    figure.colorbar(image, ax=ax, shrink=0.8).set_label(label, fontsize=10)
    return figure


def picture(rgb):
    figure = _figure(4, 4)
    figure.add_subplot().imshow(rgb)
    return figure


def gradcam_stats(analysis):
    figure = _figure(10, 8)
    (hist_ax, profile_ax), (region_ax, text_ax) = figure.subplots(2, 2)
    values = analysis.masked("gradcam")
    inside = values[analysis.mask > 0]
    layer = analysis.layers["gradcam"]

    hist_ax.hist(inside, bins=50, alpha=0.7, color="#3498db", edgecolor="black")
    hist_ax.set_xlabel("Grad-CAM++ value", fontsize=10)
    hist_ax.set_ylabel("Brain pixels", fontsize=10)
    hist_ax.set_title(f"Value Distribution in the Brain\n(Layer: {layer})", fontsize=10)
    hist_ax.grid(True, alpha=0.3)

    profile_ax.plot(values[values.shape[0] // 2, :], color="#e74c3c", linewidth=2)
    profile_ax.set_xlabel("Column", fontsize=10)
    profile_ax.set_ylabel("Grad-CAM++ value", fontsize=10)
    profile_ax.set_title("Profile Along the Center Row", fontsize=10)
    profile_ax.grid(True, alpha=0.3)

    h, w = values.shape
    regions = {
        "Upper left": values[: h // 2, : w // 2],
        "Upper right": values[: h // 2, w // 2 :],
        "Lower left": values[h // 2 :, : w // 2],
        "Lower right": values[h // 2 :, w // 2 :],
    }
    region_ax.bar(
        list(regions),
        [r.mean() for r in regions.values()],
        color=["#9b59b6", "#f39c12", "#2ecc71", "#e74c3c"],
        alpha=0.7,
    )
    region_ax.set_ylabel("Mean value", fontsize=10)
    region_ax.set_title("Mean Value in Each Image Quadrant", fontsize=11, fontweight="bold")
    region_ax.tick_params(axis="x", rotation=30)
    region_ax.grid(True, alpha=0.3)

    peak = np.unravel_index(np.argmax(values), values.shape)
    text = (
        "GRAD-CAM++ STATISTICS\n\n"
        f"Layer: {layer}\n\n"
        f"Max:     {values.max():.4f}\n"
        f"Mean:    {inside.mean() if inside.size else 0:.4f} (brain)\n"
        f"Std dev: {inside.std() if inside.size else 0:.4f} (brain)\n"
        f"Median:  {np.median(inside) if inside.size else 0:.4f} (brain)\n\n"
        f"Peak (row, col):  ({peak[0]}, {peak[1]})\n"
        f"Pixels >= 0.5:    {int((values >= 0.5).sum())}\n"
        f"Spread (>= 0.1):  {analysis.spread():.1%} of the brain"
    )
    _text_panel(text_ax, text)
    return figure


def guided_stats(analysis):
    figure = _figure(10, 8)
    (contour_ax, gradient_ax), (scatter_ax, text_ax) = figure.subplots(2, 2)
    guided = analysis.masked("guided_gradcam")
    gradcam = analysis.masked("gradcam")
    layer = analysis.layers["guided_gradcam"]

    step = max(1, guided.shape[0] // 20)
    rows, cols = np.mgrid[0 : guided.shape[0] : step, 0 : guided.shape[1] : step]
    contours = contour_ax.contour(cols, rows, guided[::step, ::step], levels=10, cmap="viridis")
    contour_ax.clabel(contours, inline=True, fontsize=8)
    contour_ax.invert_yaxis()  # image orientation: row 0 at the top
    contour_ax.set_aspect("equal")
    contour_ax.set_title(f"Contours\n(Layer: {layer})", fontsize=10)
    contour_ax.grid(True, alpha=0.3)

    gy, gx = np.gradient(guided)
    magnitude = np.hypot(gx, gy)
    image = gradient_ax.imshow(magnitude, cmap="plasma")
    gradient_ax.set_title("Spatial Gradient Magnitude", fontsize=11, fontweight="bold")
    figure.colorbar(image, ax=gradient_ax, shrink=0.8).set_label("Gradient", fontsize=10)

    scatter_ax.scatter(gradcam.ravel(), guided.ravel(), alpha=0.5, s=1, color="#e74c3c")
    scatter_ax.set_xlabel("Grad-CAM++", fontsize=10)
    scatter_ax.set_ylabel("Guided Grad-CAM++", fontsize=10)
    scatter_ax.set_title("Pixel Values of the Two Methods", fontsize=11, fontweight="bold")
    scatter_ax.grid(True, alpha=0.3)
    both_vary = gradcam.std() > 0 and guided.std() > 0
    correlation = float(np.corrcoef(gradcam.ravel(), guided.ravel())[0, 1]) if both_vary else float("nan")
    scatter_ax.text(0.05, 0.95, f"r = {correlation:.3f}", transform=scatter_ax.transAxes, va="top", bbox=TEXT_BOX)

    text = (
        "GUIDED GRAD-CAM++ STATISTICS\n\n"
        f"Layer: {layer}\n\n"
        f"Max:       {guided.max():.4f}\n"
        f"Mean:      {guided.mean():.4f}\n"
        f"Std dev:   {guided.std():.4f}\n"
        f"Below 0.01: {(guided < 0.01).mean():.1%} of pixels\n\n"
        f"Max gradient:  {magnitude.max():.4f}\n"
        f"Mean gradient: {magnitude.mean():.4f}\n\n"
        f"Pearson r with Grad-CAM++: {correlation:.3f}"
    )
    _text_panel(text_ax, text)
    return figure


def summary(analysis):
    figure = _figure(8, 10)
    ax = figure.add_subplot()
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        summary_text(analysis),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        fontfamily="monospace",
        bbox={**TEXT_BOX, "facecolor": "white"},
    )
    return figure


def placeholder():
    figure = Figure(figsize=(4, 3), facecolor="#f0f0f0")
    ax = figure.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "Maximize to view\nthis chart",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "beige", "alpha": 0.9},
    )
    return figure


def build_all(analysis):
    """Figure for each pane title in PANES."""
    figures = {
        "Original Slice": original_slice(analysis),
        "Prediction": prediction(analysis),
        "Class Maps": class_maps(analysis),
        "Class Probabilities": class_probabilities(analysis),
    }
    for key, name in MAPS.items():
        label = "Mean relevance" if key == "consensus" else "Relevance"
        figures[f"{name} Raw"] = heatmap(analysis.maps[key], label)
        figures[f"{name} Masked"] = heatmap(analysis.masked(key), label)
        figures[f"{name} Overlay"] = picture(analysis.overlay(key))
    figures["Grad-CAM++ Stats"] = gradcam_stats(analysis)
    figures["Guided Grad-CAM++ Stats"] = guided_stats(analysis)
    figures["Summary"] = summary(analysis)
    return {title: figures[title] for title, _ in PANES}
