import numpy as np
from tensorflow import keras

from neuroxai.model_check import check_model


def classifier(backbone=None, classes=3, size=224):
    inputs = keras.Input((size, size, 3))
    if backbone is None:
        features = keras.layers.Conv2D(4, 3, strides=4, activation="relu", name="conv")(inputs)
    else:
        features = backbone(inputs)
    pooled = keras.layers.GlobalAveragePooling2D()(features)
    return keras.Model(inputs, keras.layers.Dense(classes, activation="softmax")(pooled))


def test_a_small_cnn_has_an_unknown_backbone():
    info = check_model(classifier())
    assert info.errors == []
    assert info.backbone is None and len(info.warnings) == 1
    assert info.default_layer == "conv"
    batch = np.full((1, 224, 224, 3), 255.0, dtype=np.float32)
    np.testing.assert_array_equal(info.preprocess(batch.copy()), batch)  # values are not scaled


def test_wrong_class_count_and_input_size_are_errors():
    info = check_model(classifier(classes=2, size=64))
    assert len(info.errors) == 2


def test_mobilenet_v2_backbone_and_its_input_scaling():
    backbone = keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    info = check_model(classifier(backbone))
    assert info.backbone == "MobileNetV2"
    assert info.errors == [] and info.warnings == []
    assert info.layers == [backbone.name]
    scaled = info.preprocess(np.array([[0.0, 127.5, 255.0]], dtype=np.float32))
    np.testing.assert_allclose(scaled, [[-1.0, 0.0, 1.0]])
