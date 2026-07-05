"""
Training script — two-stage transfer learning:

    Stage 1: Freeze the backbone and train only the newly added head
                     (GAP + Dense layers). This is fast and prevents the randomly
                     initialized head from corrupting the pretrained backbone weights.

    Stage 2: Unfreeze top backbone layers (fine-tuning) and train the whole
                     model with a very small learning rate so it adapts to X-ray data.

Usage:
        export PNEUMONIA_DATA_DIR=/path/to/chest_xray
        python train.py
"""

import os
import tensorflow as tf

import config
import sys
from data_pipeline import get_datasets, compute_class_weights
from model import build_model, compile_model, unfreeze_top_layers


def main():
    tf.random.set_seed(config.SEED)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # Quick existence check so users get a clear error if dataset not found
    if not os.path.exists(config.TRAIN_DIR):
        print(f"ERROR: Expected training data at: {config.TRAIN_DIR}")
        print("Make sure the dataset is downloaded and/or set the PNEUMONIA_DATA_DIR environment variable.")
        sys.exit(1)

    print("=== Loading data ===")
    train_ds, val_ds, test_ds, class_names = get_datasets()
    class_weights = compute_class_weights(config.TRAIN_DIR)

    print(f"\n=== Building model: {config.BACKBONE} ===")
    model, base = build_model()
    compile_model(model, config.LEARNING_RATE_STAGE1)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.BEST_MODEL_PATH, monitor="val_auc",
            mode="max", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]

    print("\n=== Stage 1: Train head (backbone frozen) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_STAGE1,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\n=== Stage 2: Fine-tuning (unfreeze top backbone layers) ===")
    unfreeze_top_layers(base, n_layers=30)
    compile_model(model, config.LEARNING_RATE_STAGE2)  # very small LR

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_STAGE2,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(config.FINAL_MODEL_PATH)
    print(f"\nModel saved to: {config.FINAL_MODEL_PATH}")

    print("\n=== Evaluation on test dataset ===")
    results = model.evaluate(test_ds, return_dict=True)
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
