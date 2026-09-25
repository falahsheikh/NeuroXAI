"""Checks that a Keras model can be used by the Analysis Tool, and the input scaling that it needs."""

from dataclasses import dataclass, field

from tensorflow import keras

from .config import CLASS_NAMES, IMG_SIZE
from .xai import candidate_layers, feature_layer_name, walk

# Layer names that identify the Keras application backbones, as in eAlz (github.com/falahsheikh/eAlz).
_SIGNATURES = {
    "EfficientNetV2B0": {"stem_conv", "top_conv"},
    "MobileNetV2": {"Conv_1", "out_relu"},
    "DenseNet121": {"conv5_block16_concat"},
}
# Each backbone was trained with the input scaling of its Keras application.
# The EfficientNet models scale their input themselves, so their function returns the input unchanged.
_PREPROCESS = {
    "EfficientNetV2B0": keras.applications.efficientnet_v2.preprocess_input,
    "MobileNetV2": keras.applications.mobilenet_v2.preprocess_input,
    "DenseNet121": keras.applications.densenet.preprocess_input,
}


def _unchanged(x):
    return x


def detect_backbone(model):
    """Name of the Keras application backbone of the model, or None if it is not known."""
    names = {layer.name for layer in walk(model)}
    for backbone, signature in _SIGNATURES.items():
        if signature <= names and not (backbone == "DenseNet121" and "conv5_block17_concat" in names):
            return backbone
    return None


@dataclass
class ModelInfo:
    """Properties of a model. The Analysis Tool cannot use a model with errors."""

    backbone: str | None
    parameters: int
    num_classes: int
    input_shape: tuple
    layers: list = field(default_factory=list)
    default_layer: str | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def preprocess(self):
        """Function that scales a batch of RGB 0-255 images for the model."""
        return _PREPROCESS.get(self.backbone, _unchanged)

    def describe(self):
        backbone = self.backbone or "unknown (input values 0-255, not scaled)"
        return (
            f"Backbone: {backbone}\n"
            f"Parameters: {self.parameters:,}\n"
            f"Classes: {self.num_classes}\n"
            f"Input: {' x '.join(str(s) for s in self.input_shape)}\n"
            f"Layers for Grad-CAM++: {len(self.layers)} (default: {self.default_layer})"
        )


def check_model(model):
    """ModelInfo for a model: errors if it is not a 224x224 RGB CN/EMCI/LMCI CNN classifier."""
    input_shape = tuple(model.inputs[0].shape[1:])
    info = ModelInfo(
        backbone=detect_backbone(model),
        parameters=model.count_params(),
        num_classes=model.outputs[0].shape[-1],
        input_shape=input_shape,
        layers=candidate_layers(model),
    )
    if info.num_classes != len(CLASS_NAMES):
        info.errors.append(
            f"The model has {info.num_classes} outputs. NeuroXAI needs {len(CLASS_NAMES)}: {', '.join(CLASS_NAMES)}."
        )
    if input_shape != (*IMG_SIZE, 3):
        info.errors.append(f"The model input is {input_shape}. NeuroXAI gives {IMG_SIZE[0]}x{IMG_SIZE[1]} RGB images.")
    if not info.layers:
        info.errors.append("The model has no 2D convolution layer. Grad-CAM++ needs one.")
    else:
        feature = feature_layer_name(model)
        info.default_layer = feature if feature in info.layers else info.layers[-1]
    if info.backbone is None:
        info.warnings.append(
            "The backbone is not known. NeuroXAI gives the model pixel values from 0 to 255. "
            "If the model was trained with other input scaling, its results are not correct."
        )
    return info
