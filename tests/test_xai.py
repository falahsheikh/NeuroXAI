import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from neuroxai import xai


def small_cnn(seed=0):
    keras.utils.set_random_seed(seed)
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

    result = xai.gradcam_plus_plus(model, image, target, "conv")
    assert result.shape == (8, 8)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    # The default layer is the input of the global average pooling, here the same layer.
    np.testing.assert_allclose(xai.gradcam_plus_plus(model, image, target), result)


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
    expected = (w1 @ (upstream * (upstream > 0))).reshape(2, 2, 1)

    np.testing.assert_allclose(xai.guided_backprop(model, image, 0), expected, rtol=1e-6)


def nested_and_flat_models():
    """The same weights as a model with a nested backbone and as a flat model."""
    keras.utils.set_random_seed(3)
    backbone_input = keras.Input((8, 8, 3))
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu")(backbone_input)
    backbone = keras.Model(
        backbone_input, keras.layers.Conv2D(6, 3, padding="same", activation="relu")(x), name="backbone"
    )
    inputs = keras.Input((8, 8, 3))
    pooled = keras.layers.GlobalAveragePooling2D()(backbone(inputs))
    nested = keras.Model(inputs, keras.layers.Dense(3, activation="softmax")(pooled))

    flat_inputs = keras.Input((8, 8, 3))
    y = keras.layers.Conv2D(4, 3, padding="same", activation="relu")(flat_inputs)
    y = keras.layers.Conv2D(6, 3, padding="same", activation="relu", name="features")(y)
    flat = keras.Model(
        flat_inputs, keras.layers.Dense(3, activation="softmax")(keras.layers.GlobalAveragePooling2D()(y))
    )
    flat.set_weights(nested.get_weights())
    return nested, flat


def test_nested_backbones_give_the_maps_of_the_flat_model():
    nested, flat = nested_and_flat_models()
    image = np.random.default_rng(2).normal(size=(1, 8, 8, 3)).astype("float32")
    assert xai.candidate_layers(nested) == ["backbone"]
    assert xai.feature_layer_name(flat) == "features"
    np.testing.assert_allclose(
        xai.gradcam_plus_plus(nested, image, 2), xai.gradcam_plus_plus(flat, image, 2), rtol=1e-5, atol=1e-6
    )
    # The guided copy must also change the rectifiers inside the nested backbone.
    np.testing.assert_allclose(
        xai.guided_backprop(nested, image, 2), xai.guided_backprop(flat, image, 2), rtol=1e-5, atol=1e-6
    )


def test_consensus_mask_and_overlay():
    a = np.array([[0.0, 1.0], [2.0, 4.0]])
    np.testing.assert_allclose(xai.consensus([a, 2 * a + 1]), a / 4)
    np.testing.assert_array_equal(xai.consensus([np.ones((2, 2))]), np.zeros((2, 2)))

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = 200
    mask = xai.brain_mask(image)
    assert mask[10, 10] == 1 and mask[0, 0] == 0
    blended = xai.overlay(np.zeros((20, 20)), image, alpha=0.5)
    assert blended.shape == image.shape and blended.dtype == np.uint8


def test_explainer_selects_layers():
    explainer = xai.Explainer(small_cnn())
    assert explainer.layers == ["conv"]
    assert explainer.selected == {"gradcam": "conv", "guided_gradcam": "conv"}
    with pytest.raises(ValueError):
        explainer.set_layer("gradcam", "dense")
    with pytest.raises(ValueError):
        explainer.set_layer("consensus", "conv")
    image = np.random.default_rng(0).normal(size=(1, 8, 8, 3)).astype("float32")
    guided = explainer.guided_gradcam(image, 0)
    assert guided.shape == (8, 8) and 0 <= guided.min() and guided.max() <= 1


def test_models_without_convolutions_are_rejected():
    inputs = keras.Input((4,))
    with pytest.raises(ValueError):
        xai.Explainer(keras.Model(inputs, keras.layers.Dense(3)(inputs)))
