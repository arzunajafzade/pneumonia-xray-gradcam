"""
Configuration parameters for the project — Multimodal AI for Medical Imaging
(Chest X-Ray Pneumonia Detection)
"""

import os

# ---------- Paths ----------
# Root folder of the "Chest X-Ray Images (Pneumonia)" dataset (from Kaggle).
# Expected directory structure:
#   DATA_DIR/train/NORMAL/*.jpeg
#   DATA_DIR/train/PNEUMONIA/*.jpeg
#   DATA_DIR/val/NORMAL/*.jpeg
#   DATA_DIR/val/PNEUMONIA/*.jpeg
#   DATA_DIR/test/NORMAL/*.jpeg
#   DATA_DIR/test/PNEUMONIA/*.jpeg
def _find_data_dir():
    # 1) Honor explicit environment variable
    env = os.environ.get("PNEUMONIA_DATA_DIR")
    if env and os.path.exists(env):
        return env

    # 2) Common project-relative locations to try
    candidates = [
        "./chest_xray",
        "./data/chest_xray",
        "./data/chest-xray",
        "./dataset/chest_xray",
        "./datasets/chest_xray",
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    # 3) Try to discover a directory named 'chest_xray' somewhere in repo root
    repo_root = os.getcwd()
    for name in os.listdir(repo_root):
        candidate = os.path.join(repo_root, name)
        if os.path.isdir(candidate) and name.lower().replace('-', '_') == 'chest_xray':
            return os.path.abspath(candidate)

    # 4) Fallback to default relative path (may not exist) so callers can handle
    return os.path.abspath("./chest_xray")


DATA_DIR = _find_data_dir()

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_SAVE_DIR = "./models"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "final_model.keras")

# ---------- Image parameters ----------
IMG_SIZE = (224, 224)   # Standard input size for ResNet50 / EfficientNetB0
CHANNELS = 3            # X-rays are grayscale; convert to 3 channels for transfer learning

# ---------- Training parameters ----------
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10      # Train only the head (backbone frozen)
EPOCHS_STAGE2 = 15      # Fine-tuning: unfreeze top backbone layers
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5

# ---------- Model selection ----------
# Choose either "resnet50" or "efficientnet"
BACKBONE = "efficientnet"

# Class names (data generator will infer these, but keep here for clarity)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ---------- Random seed ----------
SEED = 42
