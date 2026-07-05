"""
Transfer learning model — binary classifier (NORMAL vs PNEUMONIA)
built on ResNet50 or EfficientNetB0 backbone.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import config


def build_model(backbone_name=None, input_shape=None):
    backbone_name = backbone_name or config.BACKBONE
    input_shape = input_shape or (*config.IMG_SIZE, config.CHANNELS)

    if backbone_name == "resnet50":
        base = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input
    elif backbone_name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base.trainable = False  # Stage 1: freeze the backbone

    inputs = layers.Input(shape=input_shape)
    # Note: data_pipeline.py normalizes images to [0,1]. Keras' preprocess_input
    # functions expect [0,255], so we rescale back to 0-255 here before
    # calling the backbone-specific preprocess function.
    x = layers.Rescaling(255.0)(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs, outputs, name=f"pneumonia_{backbone_name}")
    return model, base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_layers(base_model, n_layers=30):
    """Unfreeze the top `n_layers` of the backbone for fine-tuning (Stage 2).

    Fully unfreezing the backbone can increase overfitting risk on small
    medical datasets, so we only unfreeze the top layers (which learn more
    specific features).
    """
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    return base_model
