"""
Grad-CAM (Gradient-weighted Class Activation Mapping)

Provides a visual explanation for why the model predicts "PNEUMONIA" or
"NORMAL" for a given image. It computes a heatmap from the backbone's last
feature map (include_top=False) and the prediction gradients, then overlays
that heatmap on the original X-ray to show where the model is focusing.

In medical AI this is important: high accuracy alone is not enough — a
radiologist should see where the model looked to judge whether the
prediction is trustworthy.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _build_grad_model(model):
    """Build an auxiliary model that returns (last_conv_output, final_prediction).

    Note (Keras 3): the backbone (ResNet50/EfficientNet) is nested inside
    the outer model (see model.py where `x = base(x, training=False)`). In
    Keras 3 directly referencing `.output` of nested models can cause the
    "Output with path `0` is not connected to `inputs`" error due to node
    differences in the underlying graph.

    Workaround: re-call the outer model layers from the model input forward
    to create a fresh computation graph connected to the inputs while
    preserving trained weights.
    """
    inp = model.input
    h = inp
    conv_output = None
    for layer in model.layers[1:]:
        h = layer(h)
        if isinstance(layer, tf.keras.Model):
            conv_output = h  # backbone output = last convolutional feature map

    if conv_output is None:
        raise ValueError("Backbone (nested Model) output not found in the model.")

    return tf.keras.models.Model(inp, [conv_output, h])


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    img_array: shape (1, H, W, 3) — normalized the same way as in
               data_pipeline.py (the model may also apply preprocess_input).
    model: full Keras model (as returned by build_model())

    Returns: (heatmap in [0,1], class probability)
    """
    grad_model = _build_grad_model(model)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(predictions.numpy()[0][0])


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """original_img: (H, W, 3), values may be in [0,1] or [0,255]."""
    if original_img.max() <= 1.0:
        original_img_255 = (original_img * 255).astype(np.uint8)
    else:
        original_img_255 = original_img.astype(np.uint8)

    heatmap_resized = np.uint8(255 * heatmap)
    # Some environments may have an unexpected matplotlib build where
    # `matplotlib.cm.get_cmap` is not present. Try a few fallbacks so
    # this function works more robustly across setups.
    try:
        jet = cm.get_cmap("jet")
    except Exception:
        try:
            jet = plt.get_cmap("jet")
        except Exception:
            # final fallback: use 'viridis' if 'jet' unavailable
            jet = plt.get_cmap("viridis")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]

    jet_heatmap_img = tf.keras.utils.array_to_img(jet_heatmap * 255)
    jet_heatmap_img = jet_heatmap_img.resize(
        (original_img_255.shape[1], original_img_255.shape[0])
    )
    jet_heatmap_arr = tf.keras.utils.img_to_array(jet_heatmap_img)

    superimposed = jet_heatmap_arr * alpha + original_img_255
    superimposed = tf.keras.utils.array_to_img(superimposed)
    return superimposed


def explain_prediction(model, img_array, original_img, class_names,
                        save_path=None):
    """Full workflow: prediction + Grad-CAM + save visualization to file."""
    heatmap, prob = make_gradcam_heatmap(img_array, model)

    pred_class = class_names[1] if prob >= 0.5 else class_names[0]
    confidence = prob if prob >= 0.5 else 1 - prob

    superimposed = overlay_heatmap(original_img, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    display_original = (original_img.astype("uint8")
                         if original_img.max() > 1 else original_img)
    axes[0].imshow(display_original)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(superimposed)
    axes[1].set_title(f"Grad-CAM: {pred_class} ({confidence:.1%})")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Result saved: {save_path}")
    plt.close(fig)

    return pred_class, confidence
