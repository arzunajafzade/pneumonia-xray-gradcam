"""
Run a prediction on a single chest X-ray and produce a Grad-CAM explanation.

Usage:
    python predict.py --image path/to/xray.jpeg
"""

import argparse
import numpy as np
import tensorflow as tf

import config
from gradcam import explain_prediction


def load_and_preprocess(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    original = tf.keras.utils.img_to_array(img)          # [0,255], (H,W,3)
    normalized = original / 255.0                         # same normalization as data_pipeline
    batch = np.expand_dims(normalized, axis=0)             # (1,H,W,3)
    return batch, original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the X-ray image")
    parser.add_argument("--model", default=config.BEST_MODEL_PATH)
    parser.add_argument("--output", default="gradcam_result.png")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    img_batch, original_img = load_and_preprocess(args.image)

    pred_class, confidence = explain_prediction(
        model, img_batch, original_img, config.CLASS_NAMES,
        save_path=args.output,
    )

    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Visual explanation: {args.output}")


if __name__ == "__main__":
    main()
