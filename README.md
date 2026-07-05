# Multimodal AI for Medical Imaging — Pneumonia Detection

A project that detects pneumonia from chest X-ray images using transfer
learning (ResNet50 / EfficientNetB0) and provides Grad-CAM explainability.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Download from Kaggle: **Chest X-Ray Images (Pneumonia)**
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Unzip into a folder with the following structure:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Set the dataset path via environment variable:

```bash
export PNEUMONIA_DATA_DIR=/full/path/to/chest_xray
```

## Training

```bash
python train.py
```

This runs two-stage training (see `train.py` for details):
1. Train head (backbone frozen)
2. Fine-tuning (unfreeze top backbone layers)

The best model will be saved to `models/best_model.keras`.

## Prediction + Grad-CAM explanation

```bash
python predict.py --image path/to/xray.jpeg --output result.png
```

This will save the prediction (NORMAL/PNEUMONIA) and a heatmap showing
which areas of the image influenced the model's decision.

## File structure

| File | Description |
|---|---|
| `config.py` | All hyperparameters and paths |
| `data_pipeline.py` | Data loading, augmentation, class balancing |
| `model.py` | Transfer learning model implementation |
| `train.py` | Two-stage training script |
| `gradcam.py` | Grad-CAM (explainability) implementation |
| `predict.py` | Single-image inference + explanation |

## Notes

Change the backbone selection in `config.py` by setting `BACKBONE` to
`"resnet50"` or `"efficientnet"`.
