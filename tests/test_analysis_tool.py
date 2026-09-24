from pathlib import Path

import cv2
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import analysis_tool as at

MODEL = Path(__file__).resolve().parents[1] / "app" / "analysis" / "TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras"


def small_cnn():
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8, 8, 3))
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="conv")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(5, activation="relu")(x)
    return keras.Model(inputs, keras.layers.Dense(3, activation="softmax")(x))


def test_gradcam_plus_plus_matches_reference_equation():
    model = small_cnn()
    image = np.random.default_rng(1).normal(size=(1, 8, 8, 3)).astype("float32")
    target = 1

    # Reference: Chattopadhay et al. (2018), Eq. 19 with the exponential score, written in NumPy.
    head = model.layers[-1]
    features = keras.Model(model.input, [model.get_layer("conv").output, head.input])
    with tf.GradientTape() as tape:
        activations, pooled = features(image)
        logit = (tf.matmul(pooled, head.kernel) + head.bias)[:, target]
    grads = tape.gradient(logit, activations)[0].numpy()
    activations = activations[0].numpy()
    denominator = 2 * grads**2 + activations.sum(axis=(0, 1)) * grads**3
    denominator[denominator == 0] = 1
    weights = (grads**2 / denominator * np.maximum(grads, 0)).sum(axis=(0, 1))
    expected = np.maximum((activations * weights).sum(axis=-1), 0)
    expected /= expected.max()
    expected = tf.image.resize(expected[..., None], at.IMG_SIZE, method="bilinear")[..., 0].numpy()

    result = at.XAIProcessor(model).make_gradcam_plus_plus(image, target, "conv")
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("relu_as_layer", [False, True])
def test_guided_backprop_matches_hand_derivation(relu_as_layer):
    # x (4 values) -> z = x W1 -> ReLU -> logits = h W2. Guided backprop keeps a hidden unit only
    # where z > 0 and the gradient arriving from W2 is positive.
    inputs = keras.Input((2, 2, 1))
    x = keras.layers.Flatten()(inputs)
    if relu_as_layer:
        x = keras.layers.ReLU(6.0)(keras.layers.Dense(3, use_bias=False, name="hidden")(x))
    else:
        x = keras.layers.Dense(3, activation="relu", use_bias=False, name="hidden")(x)
    model = keras.Model(inputs, keras.layers.Dense(2, activation="softmax", use_bias=False, name="out")(x))

    w1 = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -1.0], [-0.5, 0.5, 1.0], [1.0, 1.0, 1.0]], dtype="float32")
    w2 = np.array([[1.0, -1.0], [-2.0, 1.0], [0.5, 0.5]], dtype="float32")
    model.get_layer("hidden").set_weights([w1])
    model.get_layer("out").set_weights([w2])
    image = np.ones((1, 2, 2, 1), dtype="float32")

    upstream = w2[:, 0]  # [1.0, -2.0, 0.5]; every hidden unit is active for this input
    guided = np.abs(w1 @ (upstream * (upstream > 0))).reshape(2, 2)

    result = at.XAIProcessor(model).make_guided_backprop(image, 0)
    np.testing.assert_allclose(result, guided, rtol=1e-6)


def test_bundled_model_analysis():
    model = keras.models.load_model(MODEL)
    assert model.count_params() == 6_083_667
    processor = at.XAIProcessor(model)
    assert processor.get_layer_for_method("gradcam") == "top_conv"

    # Synthetic slice, preprocessed like MedicalXAIInterface.get_img_array.
    rows, cols = np.indices((256, 256))
    brain = ((rows - 128) / 100.0) ** 2 + ((cols - 128) / 80.0) ** 2 < 1
    slice_data = brain * np.random.default_rng(0).uniform(40, 200, size=brain.shape)
    processed = at.MedicalImageProcessor.crop_and_resize_slice(slice_data, target_size=at.IMG_SIZE)
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB).astype("float32")
    batch = keras.applications.efficientnet_v2.preprocess_input(rgb[np.newaxis])

    probs = model.predict(batch, verbose=0)[0]
    np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-5)
    target = int(np.argmax(probs))
    maps = [processor.make_gradcam_plus_plus(batch, target), processor.make_guided_gradcam_plus_plus(batch, target)]
    maps.append(processor.create_consensus_map(maps))
    for heatmap in maps:
        assert heatmap.shape == at.IMG_SIZE
        assert np.isfinite(heatmap).all() and heatmap.min() >= 0 and heatmap.max() <= 1 + 1e-6


def test_slice_viewer_imports():
    import slice_viewer

    assert hasattr(slice_viewer, "SlicerApp")
    assert not hasattr(slice_viewer, "XAIProcessor")
