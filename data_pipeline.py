"""
Data pipeline — utilities to load, normalize and augment Chest X-Ray images.

Dataset imbalance (more pneumonia samples than normal) is common; we
compute class weights and pass them to the training loop to compensate.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import config


def _augmentation_layers():
    """Augmentation layers applied only to the TRAIN dataset.

    Vertical/horizontal flips may change clinical meaning in medical images,
    so we only apply light, clinically-plausible transforms (small
    rotation, zoom, contrast, translation).
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")


def _load_dataset(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",          # PNEUMONIA=1, NORMAL=0 (alphabetical order)
        class_names=None,
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
    )


def get_datasets():
    """Return Train / Validation / Test datasets as tf.data.Dataset objects."""
    train_ds = _load_dataset(config.TRAIN_DIR, shuffle=True)
    val_ds = _load_dataset(config.VAL_DIR, shuffle=False)
    test_ds = _load_dataset(config.TEST_DIR, shuffle=False)

    class_names = train_ds.class_names  # ['NORMAL', 'PNEUMONIA']
    print(f"Class names: {class_names}")

    augment = _augmentation_layers()
    normalization = layers.Rescaling(1.0 / 255.0)

    # Train: augmentation + normalization
    train_ds = train_ds.map(
        lambda x, y: (normalization(augment(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Val/Test: only normalization (no augmentation — for real evaluation)
    val_ds = val_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_dir):
    """Compute class weights to compensate dataset imbalance.

    In this dataset PNEUMONIA examples typically outnumber NORMAL ones.
    Without class weights the model may trivially predict the majority
    class and still show high accuracy.
    """
    import os

    counts = {}
    for idx, cls in enumerate(sorted(os.listdir(train_dir))):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            counts[idx] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

    total = sum(counts.values())
    n_classes = len(counts)
    class_weights = {
        idx: total / (n_classes * count) for idx, count in counts.items()
    }
    print(f"Image counts per class: {counts}")
    print(f"Computed class_weight: {class_weights}")
    return class_weights
