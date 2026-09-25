"""Grad-CAM++, guided backpropagation and the maps that the Analysis Tool shows.

Both methods differentiate the class score before the softmax (the logit), as defined in
Grad-CAM (Selvaraju et al., 2017) and Grad-CAM++ (Chattopadhay et al., 2018).

The functions accept flat models and models with one nested backbone model. For a nested
backbone, the layers before and after the backbone must form a chain, and Grad-CAM++ uses the
output of the backbone.
"""

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from tensorflow import keras

METHODS = {"gradcam": "Grad-CAM++", "guided_gradcam": "Guided Grad-CAM++"}

_CONV_TYPES = (keras.layers.Conv2D, keras.layers.DepthwiseConv2D, keras.layers.SeparableConv2D)
_RECTIFIERS = {"relu", "relu6", "silu", "swish"}


def walk(model):
    """All layers of model, including the layers of nested models."""
    for layer in model.layers:
        yield layer
        if isinstance(layer, keras.Model):
            yield from walk(layer)


def _backbone(model):
    return next((layer for layer in model.layers if isinstance(layer, keras.Model)), None)


def feature_layer_name(model):
    """Name of the layer that makes the final feature map (the input of the final global average pooling).

    Return None if the model has no global average pooling layer.
    """
    backbone = _backbone(model)
    if backbone is not None:
        return backbone.name
    # Search from the end: EfficientNet's squeeze-and-excitation blocks contain their own pooling layers.
    pooling = next((x for x in reversed(model.layers) if isinstance(x, keras.layers.GlobalAveragePooling2D)), None)
    if pooling is None:
        return None
    return next((x.name for x in model.layers if x.output is pooling.input), None)


def candidate_layers(model):
    """Names of the layers that Grad-CAM++ can use, in model order.

    These are the convolution layers with a spatial output larger than 1x1 and the layer that makes
    the final feature map. For a model with a nested backbone, only the backbone output can be used.
    """
    backbone = _backbone(model)
    if backbone is not None:
        return [backbone.name]
    feature = feature_layer_name(model)
    names = []
    for layer in model.layers:
        try:
            shape = layer.output.shape
        except AttributeError:  # a layer with more than one output
            continue
        spatial = len(shape) == 4 and all(size is None or size > 1 for size in shape[1:3])
        if spatial and (isinstance(layer, _CONV_TYPES) or layer.name == feature):
            names.append(layer.name)
    return names


def _forward(model, layer_name):
    """Function x -> (output of layer layer_name, input of the final layer)."""
    backbone = _backbone(model)
    if backbone is not None and layer_name == backbone.name:
        index = model.layers.index(backbone)
        before = [x for x in model.layers[:index] if not isinstance(x, keras.layers.InputLayer)]
        after = model.layers[index + 1 : -1]

        def forward(x):
            for layer in before:
                x = layer(x, training=False)
            activations = h = backbone(x, training=False)
            for layer in after:
                h = layer(h, training=False)
            return activations, h

        return forward
    split = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.layers[-1].input])
    return lambda x: split([x], training=False)  # model.inputs is a list


def _head_input(model):
    """Function x -> input of the final layer."""
    backbone = _backbone(model)
    if backbone is not None:
        forward = _forward(model, backbone.name)
        return lambda x: forward(x)[1]
    split = keras.Model(model.inputs, model.layers[-1].input)
    return lambda x: split([x], training=False)  # model.inputs is a list


def _scores(head, h):
    """Class scores before the softmax, from the input h of the final layer head."""
    if isinstance(head, keras.layers.Dense):
        logits = tf.matmul(h, tf.convert_to_tensor(head.kernel))
        return logits if head.bias is None else logits + tf.convert_to_tensor(head.bias)
    if isinstance(head, keras.layers.Activation | keras.layers.Softmax):
        return h
    return head(h, training=False)  # other final layers: use the model output


def normalize(x):
    """Divide by the maximum, so that a non-negative map is in [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    peak = x.max()
    return x / peak if peak > 0 else np.zeros_like(x)


def gradcam_plus_plus(model, image, class_index, layer_name=None):
    """Grad-CAM++ map in [0, 1] at input resolution, for one preprocessed image of shape (1, H, W, 3)."""
    layer_name = layer_name or feature_layer_name(model)
    forward = _forward(model, layer_name)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        activations, h = forward(image)
        score = _scores(model.layers[-1], h)[:, class_index]
    grads = tape.gradient(score, activations)
    if grads is None:
        raise ValueError(f"the class score does not depend on the output of layer {layer_name}")
    grads, activations = grads[0], activations[0]

    # alpha_ij^k = g^2 / (2 g^2 + sum_ab(A_ab^k) g^3), with g the gradient at (i, j) of channel k.
    grads_2, grads_3 = tf.square(grads), tf.pow(grads, 3)
    denominator = 2.0 * grads_2 + tf.reduce_sum(activations, axis=(0, 1), keepdims=True) * grads_3
    denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
    weights = tf.reduce_sum(grads_2 / denominator * tf.nn.relu(grads), axis=(0, 1))

    cam = tf.nn.relu(tf.reduce_sum(activations * weights, axis=-1))
    cam = tf.image.resize(cam[..., tf.newaxis], image.shape[1:3], method="bilinear")[..., 0]
    return normalize(cam)


def _guided(fn):
    """Wrap an activation so that its gradient passes only where the input and the incoming gradient are positive."""

    @tf.custom_gradient
    def guided(x):
        def grad(upstream):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = fn(x)
            dx = tape.gradient(y, x, output_gradients=upstream)
            return dx * tf.cast(x > 0, dx.dtype) * tf.cast(upstream > 0, dx.dtype)

        return fn(x), grad

    return guided


def _clone_layer(layer):
    if isinstance(layer, keras.layers.ReLU):  # e.g. MobileNetV2's ReLU6 layers, which have no `activation` attribute

        def relu(x, config=layer):
            return keras.activations.relu(
                x, negative_slope=config.negative_slope, max_value=config.max_value, threshold=config.threshold
            )

        return keras.layers.Activation(_guided(relu), name=layer.name)
    return layer.__class__.from_config(layer.get_config())


def guided_model(model):
    """Copy of model whose rectifying activations use the guided backpropagation rule (Springenberg et al., 2015)."""
    clone = keras.models.clone_model(model, clone_function=_clone_layer, recursive=True)
    clone.set_weights(model.get_weights())
    for layer in walk(clone):
        if getattr(getattr(layer, "activation", None), "__name__", None) in _RECTIFIERS:
            layer.activation = _guided(layer.activation)
    return clone


def guided_backprop(model, image, class_index, guided=None):
    """Guided-backpropagation gradients of the class score with respect to the input, shape (H, W, 3).

    Pass a precomputed guided_model(model) as `guided` to avoid rebuilding it for every image.
    """
    guided = guided or guided_model(model)
    head_input = _head_input(guided)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        score = _scores(guided.layers[-1], head_input(image))[:, class_index]
    return tape.gradient(score, image)[0].numpy()


def guided_gradcam_plus_plus(model, image, class_index, layer_name=None, guided=None, sigma=0.8):
    """Guided Grad-CAM++ map in [0, 1]: the guided-backprop saliency times the Grad-CAM++ map.

    The saliency is the maximum absolute gradient over the color channels. A Gaussian filter
    (standard deviation sigma pixels) smooths the product.
    """
    cam = gradcam_plus_plus(model, image, class_index, layer_name)
    saliency = np.abs(guided_backprop(model, image, class_index, guided)).max(axis=-1)
    return normalize(gaussian_filter(normalize(saliency * cam), sigma=sigma))


def consensus(maps):
    """Mean of the maps after each is scaled to [0, 1] by its minimum and maximum."""
    scaled = []
    for heatmap in maps:
        low, high = float(np.min(heatmap)), float(np.max(heatmap))
        scaled.append((heatmap - low) / (high - low) if high > low else np.zeros_like(heatmap))
    return np.mean(scaled, axis=0).astype(np.float32)


def brain_mask(image_rgb):
    """1 inside the brain and 0 outside: an Otsu threshold of the image, closed with a 3x3 kernel."""
    gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return (binary > 0).astype(np.float32)


def overlay(heatmap, image_rgb, alpha=0.6):
    """Blend a [0, 1] map (jet colors) onto an RGB uint8 image."""
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(
        cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR), 1 - alpha, colored, alpha, 0
    )
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


class Explainer:
    """Explanation maps for one model, with a selected Grad-CAM++ layer for each method."""

    def __init__(self, model):
        self.model = model
        self.layers = candidate_layers(model)
        if not self.layers:
            raise ValueError("the model has no 2D convolution layer, so Grad-CAM++ cannot be used")
        feature = feature_layer_name(model)
        self.default_layer = feature if feature in self.layers else self.layers[-1]
        self.selected = dict.fromkeys(METHODS, self.default_layer)
        self._guided = None

    def set_layer(self, method, layer_name):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; the methods are {', '.join(METHODS)}")
        if layer_name not in self.layers:
            raise ValueError(f"layer {layer_name!r} cannot be used for Grad-CAM++")
        self.selected[method] = layer_name

    def gradcam(self, image, class_index, layer_name=None):
        return gradcam_plus_plus(self.model, image, class_index, layer_name or self.selected["gradcam"])

    def guided_gradcam(self, image, class_index):
        if self._guided is None:
            self._guided = guided_model(self.model)
        return guided_gradcam_plus_plus(
            self.model, image, class_index, self.selected["guided_gradcam"], guided=self._guided
        )
