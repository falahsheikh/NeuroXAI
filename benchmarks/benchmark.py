"""Measure the time and memory that the Analysis Tool needs on this computer.

Usage: python -m benchmarks.benchmark [--model MODEL] [--image IMAGE] [--runs N] [--output FILE]

The output has the structure of results/performance_results.json, the measurements in the paper:

- model_loading: load time, growth of the process memory, file size and parameters
- inference: time of one prediction, over N runs after one warm-up run
- xai_visualization: time of one Grad-CAM++ map at the default layer, after one warm-up run
- complete_workflow: model loading + inference + Grad-CAM++ (as in the paper), and the time of one
  complete analysis of the current Analysis Tool (all maps)

Without --image, the benchmark uses a synthetic slice.
"""

import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras

from neuroxai import imaging, xai
from neuroxai.analysis.pipeline import analyze
from neuroxai.config import DEFAULT_MODEL
from neuroxai.model_check import check_model


def rss_mb():
    return psutil.Process().memory_info().rss / 2**20


def synthetic_slice():
    rows, cols = np.indices((256, 256))
    brain = ((rows - 128) / 100.0) ** 2 + ((cols - 128) / 80.0) ** 2 < 1
    gray = imaging.to_uint8(
        imaging.crop_and_resize_slice(brain * np.random.default_rng(0).uniform(40, 200, brain.shape))
    )
    return np.repeat(gray[..., None], 3, axis=-1)


def system_info():
    frequency = psutil.cpu_freq()
    return {
        "cpu_count": os.cpu_count(),
        "cpu_freq": frequency._asdict() if frequency else None,
        "memory_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "platform": sys.platform,
        "machine": platform.machine(),
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
    }


def run(model_path, image_path, runs):
    measurements = []

    tracemalloc.start()
    before, start = rss_mb(), time.perf_counter()
    model = keras.models.load_model(model_path)
    load_s = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    measurements.append(
        {
            "operation": "model_loading",
            "model_path": Path(model_path).name,
            "load_time_seconds": round(load_s, 4),
            "memory_usage_mb": round(rss_mb() - before, 2),
            "peak_memory_mb": round(peak / 2**20, 2),
            "model_size_mb": round(Path(model_path).stat().st_size / 2**20, 2),
            "model_parameters": model.count_params(),
        }
    )

    info = check_model(model)
    image = imaging.load_slice(image_path) if image_path else synthetic_slice()
    batch = info.preprocess(image.astype(np.float32)[np.newaxis])
    model.predict(batch, verbose=0)  # warm-up: the first call builds the prediction function
    times, deltas = [], []
    for _ in range(runs):
        before, start = rss_mb(), time.perf_counter()
        prediction = model.predict(batch, verbose=0)
        times.append((time.perf_counter() - start) * 1000)
        deltas.append(rss_mb() - before)
    measurements.append(
        {
            "operation": "inference",
            "num_runs": runs,
            "mean_inference_time_ms": round(float(np.mean(times)), 2),
            "std_inference_time_ms": round(float(np.std(times)), 2),
            "min_inference_time_ms": round(float(np.min(times)), 2),
            "max_inference_time_ms": round(float(np.max(times)), 2),
            "mean_memory_delta_mb": round(float(np.mean(deltas)), 2),
            "predictions_shape": list(prediction.shape),
            "sample_prediction": prediction[0].tolist(),
        }
    )

    target = int(np.argmax(prediction[0]))
    xai.gradcam_plus_plus(model, batch, target, info.default_layer)  # warm-up
    before, start = rss_mb(), time.perf_counter()
    heatmap = xai.gradcam_plus_plus(model, batch, target, info.default_layer)
    xai_ms = (time.perf_counter() - start) * 1000
    measurements.append(
        {
            "operation": "xai_visualization",
            "target_layer": info.default_layer,
            "generation_time_ms": round(xai_ms, 2),
            "memory_usage_mb": round(rss_mb() - before, 2),
            "heatmap_shape": list(heatmap.shape),
            "heatmap_stats": {"min": float(heatmap.min()), "max": float(heatmap.max()), "mean": float(heatmap.mean())},
        }
    )

    explainer = xai.Explainer(model)
    analyze(model, explainer, info.preprocess, image)  # warm-up: builds the guided-backpropagation copy
    start = time.perf_counter()
    analyze(model, explainer, info.preprocess, image)
    analysis_ms = (time.perf_counter() - start) * 1000
    measurements.append(
        {
            "operation": "complete_workflow",
            "total_time_ms": round(load_s * 1000 + float(np.mean(times)) + xai_ms, 2),
            "inference_plus_xai_ms": round(float(np.mean(times)) + xai_ms, 2),
            "complete_analysis_ms": round(analysis_ms, 2),
        }
    )
    return {"system_info": system_info(), "measurements": measurements, "timestamp": datetime.now().isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m benchmarks.benchmark", description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Keras model (default: the model of the paper)")
    parser.add_argument("--image", help="coronal slice (default: a synthetic slice)")
    parser.add_argument("--runs", type=int, default=10, help="number of timed predictions")
    parser.add_argument("--output", help="JSON file for the results (default: print them)")
    args = parser.parse_args(argv)
    results = run(args.model, args.image, args.runs)
    text = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
