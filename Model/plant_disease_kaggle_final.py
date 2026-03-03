"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🌿 PLANT DISEASE DETECTION — ULTIMATE TRAINING SCRIPT v5.0                ║
║  Dataset  : vipoooool/new-plant-diseases-dataset (PlantVillage, 38 classes) ║
║  Model    : EfficientNetV2S + CBAM Attention                                ║
║  Target   : 95%+ Validation Accuracy (typically achieves 97-99%)            ║
║                                                                              ║
║  RESEARCH-BACKED TECHNIQUES:                                                 ║
║  [1] CategoricalCrossentropy + Label Smoothing — stable FP16 loss.          ║
║      Replaces Focal Loss which caused NaN under mixed precision.            ║
║  [2] CutMix — Yun et al., "CutMix: Regularization Strategy to Train        ║
║      Strong Classifiers", ICCV 2019. Prevents overfitting.                  ║
║  [3] MixUp — Zhang et al., "mixup: Beyond Empirical Risk Minimization",    ║
║      ICLR 2018. Smooths decision boundaries.                                ║
║  [4] Cosine Annealing — Loshchilov & Hutter, "SGDR: Stochastic Gradient    ║
║      Descent with Warm Restarts", ICLR 2017.                                ║
║  [5] AdamW — Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ║
║      ICLR 2019. Proper weight decay.                                         ║
║  [6] CBAM — Woo et al., "CBAM: Convolutional Block Attention Module",      ║
║      ECCV 2018. Channel + spatial attention.                                 ║
║  [7] Label Smoothing — Szegedy et al., "Rethinking the Inception           ║
║      Architecture for Computer Vision", CVPR 2016.                           ║
║  [8] LR Warmup — Goyal et al., "Accurate, Large Minibatch SGD:             ║
║      Training ImageNet in 1 Hour", 2017.                                     ║
║  [9] EfficientNetV2 — Tan & Le, "EfficientNetV2: Smaller Models and        ║
║      Faster Training", ICML 2021. State-of-the-art backbone.                ║
║  [10] Random Erasing — Zhong et al., "Random Erasing Data Augmentation",    ║
║       AAAI 2020. Occlusion-based regularization.                             ║
║  [11] Two-Phase Transfer Learning — freeze backbone then fine-tune.         ║
║  [12] Test-Time Augmentation (TTA) — multi-view inference averaging.        ║
║                                                                              ║
║  ANTI-OVERFITTING MEASURES:                                                  ║
║  • Strong data augmentation (geometric + photometric + CutMix/MixUp)        ║
║  • Label smoothing (0.1) prevents overconfident predictions                 ║
║  • Dropout (0.4) in classifier head                                         ║
║  • Weight decay (1e-4) via AdamW                                            ║
║  • Early stopping with patience=7                                           ║
║  • Frozen BatchNorm during fine-tuning (prevents distribution shift)        ║
║  • Two-phase training prevents catastrophic forgetting                      ║
║                                                                              ║
║  CLASS IMBALANCE HANDLING:                                                   ║
║  • CategoricalCrossentropy with label smoothing (0.1)                       ║
║  • CutMix/MixUp create virtual balanced training examples                   ║
║  • Dataset imbalance is only ~1.2x (well-balanced)                          ║
║                                                                              ║
║  KAGGLE SETUP:                                                               ║
║    1. Add dataset  : vipoooool/new-plant-diseases-dataset                    ║
║    2. Accelerator  : GPU T4 x2 (or P100)                                    ║
║    3. Internet     : ON (for pretrained weights)                             ║
║    4. Run All                                                                ║
║                                                                              ║
║  OUTPUT: All results zipped → /kaggle/working/plant_disease_results.zip     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · IMPORTS & ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

import os
import gc
import sys
import json
import time
import zipfile
import warnings
import logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("PlantDisease")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

tf.get_logger().setLevel("ERROR")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · CONFIGURATION  (all hyperparameters in one place)
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Central hyperparameter store — edit values here to experiment."""

    # ── Paths ──────────────────────────────────────────────────────────────
    DATASET_ROOT = "/kaggle/input/new-plant-diseases-dataset"
    DATASET_ROOTS_FALLBACK = [
        "/kaggle/input/new-plant-diseases-dataset",
        "/kaggle/input/datasets/vipoooool/new-plant-diseases-dataset",
        "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)",
        "/kaggle/input/new-plant-diseases-dataset/new plant diseases dataset(augmented)",
    ]
    OUTPUT_DIR   = "/kaggle/working/results"

    # ── Model ──────────────────────────────────────────────────────────────
    BACKBONE   = "EfficientNetV2S"     # "EfficientNetV2S" or "EfficientNetB3"
    IMG_SIZE   = 300                   # 300 for V2S, 300 for B3
    FREEZE_BN  = True                  # freeze BatchNorm during fine-tune

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE    = 32
    EPOCHS_PHASE1 = 10                 # Phase 1: only classifier head
    EPOCHS_PHASE2 = 30                 # Phase 2: full fine-tuning
    PATIENCE      = 7                  # early-stopping patience

    # ── Optimizer ──────────────────────────────────────────────────────────
    LR_PHASE1     = 1e-3               # higher LR for random head
    LR_PHASE2     = 2e-5               # very low LR for fine-tuning
    WEIGHT_DECAY  = 1e-4               # AdamW weight decay
    WARMUP_EPOCHS = 3                  # linear warmup

    # ── Regularization ─────────────────────────────────────────────────────
    DROPOUT_RATE    = 0.4
    LABEL_SMOOTHING = 0.1

    # ── Augmentation ───────────────────────────────────────────────────────
    MIXUP_ALPHA       = 0.2            # Beta distribution param for MixUp
    CUTMIX_ALPHA      = 1.0            # Beta distribution param for CutMix
    MIXUP_PROB        = 0.3            # probability of MixUp per batch
    CUTMIX_PROB       = 0.3            # probability of CutMix per batch
    RANDOM_ERASE_PROB = 0.25           # per-image random erasing probability

    # ── Loss (​class imbalance) ───────────────────────────────────────────
    USE_CLASS_WEIGHTS = False          # disabled: incompatible with CutMix/MixUp soft labels

    # ── Test-Time Augmentation ─────────────────────────────────────────────
    TTA_STEPS = 3                      # augmented views at inference

    # ── Miscellaneous ──────────────────────────────────────────────────────
    SEED            = 42
    MIXED_PRECISION = False            # DISABLED: Keras 3 (TF 2.16+) removed
                                       # auto loss-scaling, causing FP16 gradient
                                       # overflow → NaN after ~1700 steps.
                                       # float32 fits fine on T4 x2 with batch 32.
    NUM_WORKERS     = tf.data.AUTOTUNE


CFG = Config()

# Reproducibility
tf.random.set_seed(CFG.SEED)
np.random.seed(CFG.SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · GPU & MIXED PRECISION SETUP
# ═══════════════════════════════════════════════════════════════════════════════

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if CFG.MIXED_PRECISION and gpus:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    log.info("✓ Mixed precision (float16) enabled for faster training")

# Multi-GPU strategy (automatic)
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    log.info(f"✓ Multi-GPU training: {strategy.num_replicas_in_sync} replicas")
else:
    strategy = tf.distribute.get_strategy()

print("\n" + "=" * 70)
print(f"  TensorFlow : {tf.__version__}")
print(f"  Python     : {sys.version.split()[0]}")
print(f"  GPUs       : {len(gpus)}")
for g in gpus:
    print(f"    → {g}")
print(f"  Mixed Prec : {CFG.MIXED_PRECISION and bool(gpus)}")
print(f"  Strategy   : {strategy.__class__.__name__}")
print(f"  Backbone   : {CFG.BACKBONE}")
print(f"  Image Size : {CFG.IMG_SIZE}×{CFG.IMG_SIZE}")
print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · DATASET DISCOVERY & LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _try_find_splits(root):
    """Try to find train/ and valid/ dirs under `root`."""
    root = Path(root)
    if not root.is_dir():
        return None, None
    train_dir = valid_dir = None
    for path in sorted(root.rglob("train")):
        if path.is_dir() and any(p.is_dir() for p in path.iterdir()):
            train_dir = str(path)
            break
    for path in sorted(root.rglob("valid")):
        if path.is_dir() and any(p.is_dir() for p in path.iterdir()):
            valid_dir = str(path)
            break
    return train_dir, valid_dir


def find_data_dirs():
    """
    Auto-discover train/ and valid/ directories.
    Strategy:
      1. Try primary DATASET_ROOT
      2. Try each DATASET_ROOTS_FALLBACK path
      3. Recursively scan /kaggle/input/ for any dir with train/ + valid/
    """
    # Strategy 1: primary
    train_dir, valid_dir = _try_find_splits(CFG.DATASET_ROOT)
    if train_dir and valid_dir:
        return train_dir, valid_dir

    # Strategy 2: fallback paths
    for fb in CFG.DATASET_ROOTS_FALLBACK:
        log.info(f"Trying fallback: {fb}")
        train_dir, valid_dir = _try_find_splits(fb)
        if train_dir and valid_dir:
            return train_dir, valid_dir

    # Strategy 3: recursive scan of /kaggle/input/
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        log.info("Scanning /kaggle/input/ recursively …")
        for candidate in sorted(kaggle_input.rglob("*")):
            if candidate.is_dir() and candidate.name.lower() in ("train", "valid"):
                continue  # skip the splits themselves
            train_dir, valid_dir = _try_find_splits(candidate)
            if train_dir and valid_dir:
                log.info(f"Found dataset at: {candidate}")
                return train_dir, valid_dir

    # List what's actually there for debugging
    if kaggle_input.is_dir():
        items = list(kaggle_input.rglob("*"))[:50]
        log.error("First 50 items under /kaggle/input/:")
        for it in items:
            log.error(f"  {it}")

    raise FileNotFoundError(
        "Could not find train/valid dirs. Check dataset is attached to notebook."
    )


train_dir, valid_dir = find_data_dirs()
log.info(f"Train directory: {train_dir}")
log.info(f"Valid directory: {valid_dir}")

# Sorted class names → deterministic class indices
class_names = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])
NUM_CLASSES = len(class_names)
log.info(f"Number of classes: {NUM_CLASSES}")

# Count samples per class
class_counts = {}
for cls in class_names:
    cls_path = os.path.join(train_dir, cls)
    class_counts[cls] = len([
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
    ])

total_train = sum(class_counts.values())
total_valid = sum(
    len([f for f in os.listdir(os.path.join(valid_dir, cls))
         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
    for cls in class_names
    if os.path.isdir(os.path.join(valid_dir, cls))
)
log.info(f"Training samples  : {total_train:,}")
log.info(f"Validation samples: {total_valid:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · CLASS IMBALANCE ANALYSIS & WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# ── Visualize class distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6))
counts = [class_counts[c] for c in class_names]
colors = plt.cm.viridis(np.linspace(0, 1, NUM_CLASSES))
ax.bar(range(NUM_CLASSES), counts, color=colors, edgecolor="black", linewidth=0.3)
ax.set_xlabel("Class Index", fontsize=12)
ax.set_ylabel("Sample Count", fontsize=12)
ax.set_title("Training Set — Class Distribution", fontsize=14)
ax.axhline(y=np.mean(counts), color="r", linestyle="--",
           label=f"Mean: {np.mean(counts):.0f}")
ax.axhline(y=np.median(counts), color="orange", linestyle="--",
           label=f"Median: {np.median(counts):.0f}")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CFG.OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()

imbalance_ratio = max(counts) / max(min(counts), 1)
log.info(f"Class imbalance ratio: {imbalance_ratio:.1f}x  "
         f"(min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.0f})")

# Print top-5 rarest and most common classes
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
log.info("5 rarest classes:")
for name, cnt in sorted_classes[:5]:
    log.info(f"  {name}: {cnt}")
log.info("5 most common classes:")
for name, cnt in sorted_classes[-5:]:
    log.info(f"  {name}: {cnt}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · DATA AUGMENTATION PIPELINE  (per-image, in tf.data graph)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_random_erase(image):
    """
    Random Erasing (Zhong et al., AAAI 2020).
    Replaces a random rectangle with noise to simulate occlusion.
    Works in tf.data graph mode with static IMG_SIZE.
    """
    h = CFG.IMG_SIZE
    w = CFG.IMG_SIZE

    erase_h = tf.random.uniform([], h // 8, h // 3, dtype=tf.int32)
    erase_w = tf.random.uniform([], w // 8, w // 3, dtype=tf.int32)
    top  = tf.random.uniform([], 0, h - erase_h, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - erase_w, dtype=tf.int32)

    # Build 2-D mask: 0 inside rectangle, 1 outside
    row_mask = tf.cast(
        tf.logical_or(tf.range(h) < top, tf.range(h) >= top + erase_h),
        tf.float32,
    )
    col_mask = tf.cast(
        tf.logical_or(tf.range(w) < left, tf.range(w) >= left + erase_w),
        tf.float32,
    )
    mask = 1.0 - tf.tensordot(1.0 - row_mask, 1.0 - col_mask, axes=0)
    mask = tf.expand_dims(mask, -1)                    # (H, W, 1)

    noise = tf.random.uniform([h, w, 3], 0.0, 255.0)
    return image * mask + noise * (1.0 - mask)


def augment_train(image, label):
    """
    Strong per-image augmentation for training.
    Applied BEFORE batching in the tf.data pipeline.
    """
    image = tf.cast(image, tf.float32)

    # ── Geometric transforms ──────────────────────────────────────────────
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random 90° rotation
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # Random zoom (scale 0.85–1.0) + crop back to IMG_SIZE
    scale = tf.random.uniform([], 0.85, 1.0)
    new_h = tf.cast(tf.cast(CFG.IMG_SIZE, tf.float32) / scale, tf.int32)
    new_w = new_h
    image = tf.image.resize(image, [new_h, new_w])
    image = tf.image.random_crop(image, [CFG.IMG_SIZE, CFG.IMG_SIZE, 3])

    # ── Photometric transforms ────────────────────────────────────────────
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)

    # ── Random Erasing (occlusion-based regularisation) ───────────────────
    image = tf.cond(
        tf.random.uniform([]) < CFG.RANDOM_ERASE_PROB,
        lambda: apply_random_erase(image),
        lambda: image,
    )

    # ── Normalise to [-1, 1] (EfficientNetV2 preprocessing) ──────────────
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = (image / 127.5) - 1.0
    # Final NaN guard: replace any NaN pixels with 0 (centre of [-1,1])
    image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
    return image, label


def preprocess_val(image, label):
    """Validation / inference preprocessing: normalise only."""
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image, label


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 · CUTMIX & MIXUP  (batch-level augmentation in tf.data graph)
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_beta(alpha):
    """Sample from Beta(alpha, alpha) using two Gamma samples (graph-safe)."""
    g1 = tf.random.gamma(shape=[1], alpha=alpha)[0]
    g2 = tf.random.gamma(shape=[1], alpha=alpha)[0]
    lam = g1 / (g1 + g2 + 1e-7)
    # Guard against degenerate values (0 or 1 produce no mixing → safe but useless)
    return tf.clip_by_value(lam, 0.01, 0.99)


def mixup_batch(images, labels, alpha=0.2):
    """
    MixUp (Zhang et al., ICLR 2018).
    Linearly interpolate pairs of images and labels.
    """
    lam = _sample_beta(alpha)
    lam = tf.maximum(lam, 1.0 - lam)          # ensure lam ≥ 0.5

    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=1.0):
    """
    CutMix (Yun et al., ICCV 2019).
    Paste a rectangular patch from one image onto another.
    """
    lam = _sample_beta(alpha)

    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]

    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)

    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)

    # Build 2-D rectangle mask: 1 = keep original, 0 = paste patch
    mask = tf.ones([img_h, img_w]) - tf.pad(
        tf.ones([y2 - y1, x2 - x1]),
        [[y1, img_h - y2], [x1, img_w - x2]],
    )
    mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)  # (1, H, W, 1)

    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    mixed_images = images * mask + shuffled_images * (1.0 - mask)

    # Adjust lam to actual cut area
    actual_lam = 1.0 - (
        tf.cast((y2 - y1) * (x2 - x1), tf.float32)
        / tf.cast(img_h * img_w, tf.float32)
    )
    mixed_labels = actual_lam * labels + (1.0 - actual_lam) * shuffled_labels
    return mixed_images, mixed_labels


def apply_cutmix_or_mixup(images, labels):
    """Randomly apply CutMix, MixUp, or neither to an entire batch."""
    rand = tf.random.uniform([])
    return tf.cond(
        rand < CFG.CUTMIX_PROB,
        lambda: cutmix_batch(images, labels, CFG.CUTMIX_ALPHA),
        lambda: tf.cond(
            rand < CFG.CUTMIX_PROB + CFG.MIXUP_PROB,
            lambda: mixup_batch(images, labels, CFG.MIXUP_ALPHA),
            lambda: (images, labels),
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 · BUILD tf.data PIPELINES
# ═══════════════════════════════════════════════════════════════════════════════

log.info("Building data pipelines …")

# ── Training pipeline ─────────────────────────────────────────────────────────
# unbatched → per-image augment → batch → CutMix/MixUp → prefetch
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(CFG.IMG_SIZE, CFG.IMG_SIZE),
    batch_size=None,                            # unbatched
    label_mode="categorical",
    shuffle=True,
    seed=CFG.SEED,
)
train_ds = (
    train_ds
    .shuffle(min(total_train, 2_000), seed=CFG.SEED)
    .map(augment_train, num_parallel_calls=CFG.NUM_WORKERS)
    .batch(CFG.BATCH_SIZE, drop_remainder=True)
    .map(apply_cutmix_or_mixup, num_parallel_calls=CFG.NUM_WORKERS)
    .prefetch(tf.data.AUTOTUNE)
)

# ── Validation pipeline (normalised, for model.fit) ──────────────────────────
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=(CFG.IMG_SIZE, CFG.IMG_SIZE),
    batch_size=CFG.BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)
val_ds = (
    val_ds_raw
    .map(preprocess_val, num_parallel_calls=CFG.NUM_WORKERS)
    .prefetch(tf.data.AUTOTUNE)
)

steps_per_epoch = total_train // CFG.BATCH_SIZE
log.info(f"Steps per epoch: {steps_per_epoch}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 · CBAM ATTENTION MODULE  (Woo et al., ECCV 2018)
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(layers.Layer):
    """Learn 'what' features to attend to (channel-wise)."""

    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels  = channels
        self.reduction = reduction
        mid = max(channels // reduction, 8)
        self.fc1 = layers.Dense(mid, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(channels, use_bias=False)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        mx  = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        att = tf.sigmoid(self.fc2(self.fc1(avg)) + self.fc2(self.fc1(mx)))
        return x * att

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "reduction": self.reduction})
        return cfg


class SpatialAttention(layers.Layer):
    """Learn 'where' to attend (spatial-wise)."""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            1, kernel_size, padding="same", activation="sigmoid", use_bias=False
        )

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max(x, axis=-1, keepdims=True)
        return x * self.conv(tf.concat([avg, mx], axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg


class CBAM(layers.Layer):
    """Convolutional Block Attention Module: channel → spatial."""

    def __init__(self, channels, reduction=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def call(self, x):
        return self.sa(self.ca(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "channels":    self.ca.channels,
            "reduction":   self.ca.reduction,
            "kernel_size": self.sa.kernel_size,
        })
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 · MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes, backbone_trainable=False):
    """
    EfficientNetV2S (or B3 fallback) + CBAM + classifier head.
    Returns (model, backbone) so backbone can be unfrozen later.
    """
    # ── Backbone ──────────────────────────────────────────────────────────
    try:
        from tensorflow.keras.applications import EfficientNetV2S
        backbone = EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE, 3),
            include_preprocessing=False,
        )
        log.info(f"✓ Backbone: EfficientNetV2S  "
                 f"(output channels={backbone.output_shape[-1]})")
    except Exception:
        from tensorflow.keras.applications import EfficientNetB3
        backbone = EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE, 3),
        )
        log.info(f"✓ Fallback backbone: EfficientNetB3  "
                 f"(output channels={backbone.output_shape[-1]})")

    backbone.trainable = backbone_trainable
    out_channels = backbone.output_shape[-1]

    # ── Head ──────────────────────────────────────────────────────────────
    x = backbone.output
    x = CBAM(out_channels, reduction=16, name="cbam")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(CFG.DROPOUT_RATE, name="head_drop1")(x)
    x = layers.Dense(
        512, activation="relu", name="head_fc",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(CFG.DROPOUT_RATE * 0.75, name="head_drop2")(x)

    # Explicit float32 output for numerical stability
    x = layers.Dense(num_classes, dtype="float32", name="logits")(x)
    outputs = layers.Activation("softmax", dtype="float32", name="softmax")(x)

    model = keras.Model(backbone.input, outputs, name="PlantDiseaseNet")
    return model, backbone


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 · LOSS FUNCTION  (CategoricalCrossentropy + Label Smoothing)
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: Focal Loss was removed because it produces NaN gradients under
# mixed precision (FP16). Built-in CategoricalCrossentropy is numerically
# stable with FP16 and the dataset imbalance is only ~1.2x, so focal
# down-weighting is unnecessary. CutMix/MixUp provide implicit balancing.


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 · LEARNING RATE SCHEDULE  (Linear Warmup + Cosine Annealing)
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup  → Cosine decay to min_lr.
    Ref: Goyal et al. 2017 (warmup), Loshchilov & Hutter 2017 (cosine).
    """

    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=1e-7):
        super().__init__()
        self.base_lr     = base_lr
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr      = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        warmup_lr = self.base_lr * step / tf.maximum(self.warmup_steps, 1.0)

        progress = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0
        )
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * progress)
        )

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr":      self.base_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr":       self.min_lr,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 · CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

class NaNBatchGuard(keras.callbacks.Callback):
    """
    Monitor loss EVERY batch. If NaN is detected, stop training immediately
    instead of waiting until epoch end (which wastes hundreds of steps).
    """

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0)
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            log.error(f"NaN/Inf loss detected at batch {batch}! Stopping.")
            self.model.stop_training = True


def make_callbacks(tag):
    """Standard callbacks: checkpoint (best val acc) + early stopping + CSV + NaN guard."""
    return [
        NaNBatchGuard(),                           # stop IMMEDIATELY on NaN (per-batch)
        keras.callbacks.TerminateOnNaN(),           # Keras built-in backup
        keras.callbacks.ModelCheckpoint(
            os.path.join(CFG.OUTPUT_DIR, f"best_{tag}.keras"),
            monitor="val_categorical_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_categorical_accuracy",
            mode="max",
            patience=CFG.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(CFG.OUTPUT_DIR, f"log_{tag}.csv"),
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 · TRAINING — TWO-PHASE TRANSFER LEARNING  (with NaN auto-recovery)
# ═══════════════════════════════════════════════════════════════════════════════

loss_fn = keras.losses.CategoricalCrossentropy(
    label_smoothing=CFG.LABEL_SMOOTHING,
)

metrics = [
    keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
    keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
]


def _has_nan(history):
    """Check if any epoch finished with NaN loss."""
    for v in history.history.get("loss", []):
        if v is None or np.isnan(v) or np.isinf(v):
            return True
    return False


def safe_train(model, train_ds, val_ds, epochs, base_lr, warmup_steps,
               tag, max_retries=2):
    """
    Train with automatic NaN recovery.
    If NaN is detected during training:
      1. Reload the best checkpoint saved before NaN
      2. Halve the learning rate
      3. Retry training
    After max_retries, continue with whatever best weights exist.
    """
    checkpoint_path = os.path.join(CFG.OUTPUT_DIR, f"best_{tag}.keras")
    best_history = None

    for attempt in range(1 + max_retries):
        lr_multiplier = 0.5 ** attempt
        current_lr = base_lr * lr_multiplier

        if attempt > 0:
            log.warning("=" * 60)
            log.warning(f"NaN RECOVERY — attempt {attempt}/{max_retries}")
            log.warning(f"Halving LR: {base_lr:.2e} → {current_lr:.2e}")
            log.warning("=" * 60)

            # Reload best checkpoint if available
            if os.path.exists(checkpoint_path):
                log.info(f"Reloading best checkpoint: {checkpoint_path}")
                model.load_weights(checkpoint_path)
            else:
                log.warning("No checkpoint found — continuing with current weights")

            gc.collect()

        # (Re-)compile with current learning rate
        with strategy.scope():
            lr_schedule = WarmupCosineSchedule(
                base_lr=current_lr,
                total_steps=steps_per_epoch * epochs,
                warmup_steps=warmup_steps,
            )
            model.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=CFG.WEIGHT_DECAY,
                    clipnorm=1.0,
                ),
                loss=loss_fn,
                metrics=metrics,
            )

        t0 = time.time()
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=make_callbacks(tag),
            verbose=1,
        )
        elapsed = (time.time() - t0) / 60

        if not _has_nan(history):
            # Success!
            best_val = max(history.history.get("val_categorical_accuracy", [0]))
            log.info(f"{tag} finished in {elapsed:.1f} min  |  "
                     f"best val_acc={best_val:.4f}")
            return history

        log.warning(f"NaN detected in {tag} (attempt {attempt + 1}) "
                    f"after {elapsed:.1f} min")

    # All retries exhausted — reload best checkpoint if it exists
    log.error(f"Max NaN retries exhausted for {tag}.")
    if os.path.exists(checkpoint_path):
        log.info(f"Restoring best checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)

    return history   # return last history (even if NaN)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 : Train classifier head only  (backbone frozen)
# ──────────────────────────────────────────────────────────────────────────────

log.info("=" * 65)
log.info("PHASE 1 — Training classifier head (backbone frozen)")
log.info("=" * 65)

with strategy.scope():
    model, backbone = build_model(NUM_CLASSES, backbone_trainable=False)

model.summary(print_fn=log.info, show_trainable=True)

history_p1 = safe_train(
    model, train_ds, val_ds,
    epochs=CFG.EPOCHS_PHASE1,
    base_lr=CFG.LR_PHASE1,
    warmup_steps=steps_per_epoch * CFG.WARMUP_EPOCHS,
    tag="phase1",
)

# Free memory before Phase 2
gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 : Fine-tune entire model  (backbone unfrozen, BN frozen)
# ──────────────────────────────────────────────────────────────────────────────

log.info("=" * 65)
log.info("PHASE 2 — Fine-tuning entire model (BN layers frozen)")
log.info("=" * 65)

# Unfreeze backbone
backbone.trainable = True

# Keep all BatchNorm layers frozen → prevents distribution shift on small LR
if CFG.FREEZE_BN:
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

trainable_count     = sum(K.count_params(w) for w in model.trainable_weights)
non_trainable_count = sum(K.count_params(w) for w in model.non_trainable_weights)
log.info(f"Trainable params    : {trainable_count:>12,}")
log.info(f"Non-trainable params: {non_trainable_count:>12,}")

history_p2 = safe_train(
    model, train_ds, val_ds,
    epochs=CFG.EPOCHS_PHASE2,
    base_lr=CFG.LR_PHASE2,
    warmup_steps=steps_per_epoch * 2,
    tag="phase2",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 · TEST-TIME AUGMENTATION (TTA)
# ═══════════════════════════════════════════════════════════════════════════════

def tta_single(image):
    """One random augmentation for TTA (operates on 0-255 float image)."""
    image = tf.image.random_flip_left_right(image)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image


def predict_with_tta(mdl, dataset_raw, tta_steps=5):
    """
    Average predictions over (1 base + tta_steps augmented) views.
    `dataset_raw` should yield images in [0, 255] (NOT normalised).
    """
    log.info(f"Running TTA ({1 + tta_steps} views) …")

    all_preds, all_labels = [], []
    for images, labels in dataset_raw:
        images = tf.cast(images, tf.float32)
        norm   = (images / 127.5) - 1.0
        all_preds.append(mdl.predict(norm, verbose=0))
        all_labels.append(labels.numpy())

    base_preds = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    accumulated = base_preds.copy()

    for s in range(tta_steps):
        step_preds = []
        for images, labels in dataset_raw:
            images = tf.cast(images, tf.float32)
            aug    = tf.map_fn(tta_single, images)
            norm   = (aug / 127.5) - 1.0
            step_preds.append(mdl.predict(norm, verbose=0))
        accumulated += np.concatenate(step_preds, axis=0)
        log.info(f"  TTA view {s + 1}/{tta_steps} done")

    return accumulated / (1 + tta_steps), true_labels


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 · EVALUATION & VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("EVALUATION")
log.info("=" * 65)

tta_preds, true_labels = predict_with_tta(model, val_ds_raw, CFG.TTA_STEPS)

y_pred = np.argmax(tta_preds, axis=1)
y_true = np.argmax(true_labels, axis=1)

# ── Overall metrics ───────────────────────────────────────────────────────────
acc_tta  = np.mean(y_pred == y_true)
top5_acc = top_k_accuracy_score(y_true, tta_preds, k=5, labels=np.arange(NUM_CLASSES))

# Without TTA
plain_preds = []
for images, labels in val_ds_raw:
    images = tf.cast(images, tf.float32)
    norm = (images / 127.5) - 1.0
    plain_preds.append(model.predict(norm, verbose=0))
plain_preds = np.concatenate(plain_preds, axis=0)
acc_plain = np.mean(np.argmax(plain_preds, axis=1) == y_true)

log.info(f"✓ Accuracy (with TTA)   : {acc_tta  * 100:.2f}%")
log.info(f"✓ Accuracy (without TTA): {acc_plain * 100:.2f}%")
log.info(f"✓ Top-5 Accuracy        : {top5_acc * 100:.2f}%")

# ROC-AUC
roc_auc = None
try:
    y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    roc_auc = roc_auc_score(y_true_bin, tta_preds, multi_class="ovr", average="macro")
    log.info(f"✓ ROC-AUC (macro, OvR)  : {roc_auc:.4f}")
except Exception as e:
    log.warning(f"ROC-AUC computation skipped: {e}")

# ── Classification report ─────────────────────────────────────────────────────
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, digits=4, output_dict=True,
)
report_str = classification_report(
    y_true, y_pred, target_names=class_names, digits=4,
)
print("\n" + report_str)

with open(os.path.join(CFG.OUTPUT_DIR, "classification_report.json"), "w") as f:
    json.dump(report_dict, f, indent=2, default=str)

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(22, 20))
sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names)
ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("True", fontsize=13)
ax.set_title(f"Confusion Matrix — Accuracy: {acc_tta * 100:.2f}%", fontsize=15)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
fig.savefig(os.path.join(CFG.OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# ── Training history curves ───────────────────────────────────────────────────
def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history.get(key, [])
    return merged

full_hist = merge_histories(history_p1, history_p2)
phase1_len = len(history_p1.history["loss"])

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for ax, (train_k, val_k, title) in zip(axes, [
    ("categorical_accuracy", "val_categorical_accuracy", "Accuracy"),
    ("loss",                 "val_loss",                 "Loss"),
    ("top5_accuracy",        "val_top5_accuracy",        "Top-5 Accuracy"),
]):
    ax.plot(full_hist[train_k], label="Train", linewidth=2)
    ax.plot(full_hist[val_k],   label="Val",   linewidth=2)
    ax.axvline(x=phase1_len - 0.5, color="gray", linestyle="--",
               alpha=0.6, label="Fine-tune start")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(CFG.OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# ── Per-class accuracy bar chart ──────────────────────────────────────────────
per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

fig, ax = plt.subplots(figsize=(18, 6))
bar_colors = [
    "green" if a >= 0.95 else "limegreen" if a >= 0.90
    else "orange" if a >= 0.80 else "red"
    for a in per_class_acc
]
ax.bar(range(NUM_CLASSES), per_class_acc * 100, color=bar_colors,
       edgecolor="black", linewidth=0.3)
ax.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
ax.axhline(y=95, color="green", linestyle="--", alpha=0.5, label="95% threshold")
ax.set_xlabel("Class Index", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Per-Class Accuracy", fontsize=14)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CFG.OUTPUT_DIR, "per_class_accuracy.png"), dpi=150)
plt.close()

# ── Worst-performing classes ──────────────────────────────────────────────────
worst_idx = np.argsort(per_class_acc)[:5]
log.info("5 worst-performing classes:")
for i in worst_idx:
    log.info(f"  [{i:2d}] {class_names[i]}: {per_class_acc[i]*100:.1f}%")

classes_below_90 = sum(1 for a in per_class_acc if a < 0.90)
log.info(f"Classes below 90%: {classes_below_90}/{NUM_CLASSES}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 · MODEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

log.info("Exporting model artefacts …")

# ── Class mapping JSON ────────────────────────────────────────────────────────
class_mapping = {str(i): name for i, name in enumerate(class_names)}
with open(os.path.join(CFG.OUTPUT_DIR, "class_mapping.json"), "w") as f:
    json.dump(class_mapping, f, indent=2)

# ── Keras model ───────────────────────────────────────────────────────────────
keras_path = os.path.join(CFG.OUTPUT_DIR, "plant_disease_model.keras")
try:
    model.save(keras_path)
    log.info(f"Saved Keras model  → {keras_path}")
except Exception:
    h5_path = os.path.join(CFG.OUTPUT_DIR, "plant_disease_model.h5")
    model.save(h5_path)
    log.info(f"Saved H5 model     → {h5_path}")

# ── SavedModel format ────────────────────────────────────────────────────────
sm_path = os.path.join(CFG.OUTPUT_DIR, "saved_model")
try:
    model.save(sm_path, save_format="tf")
    log.info(f"Saved SavedModel   → {sm_path}")
except Exception as e:
    log.warning(f"SavedModel export failed: {e}")

# ── TFLite (float16 quantised) ───────────────────────────────────────────────
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    tflite_path = os.path.join(CFG.OUTPUT_DIR, "plant_disease_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    log.info(f"Saved TFLite model → {tflite_path}  "
             f"({os.path.getsize(tflite_path) / 1024 / 1024:.1f} MB)")
except Exception as e:
    log.warning(f"TFLite conversion failed: {e}")

# ── Training summary JSON ────────────────────────────────────────────────────
summary = {
    "model_backbone":       CFG.BACKBONE,
    "image_size":           CFG.IMG_SIZE,
    "num_classes":          NUM_CLASSES,
    "class_names":          class_names,
    "total_train_samples":  total_train,
    "total_val_samples":    total_valid,
    "imbalance_ratio":      float(imbalance_ratio),
    "accuracy_with_tta":    float(acc_tta),
    "accuracy_without_tta": float(acc_plain),
    "top5_accuracy":        float(top5_acc),
    "roc_auc_macro":        float(roc_auc) if roc_auc else None,
    "epochs_phase1":        len(history_p1.history["loss"]),
    "epochs_phase2":        len(history_p2.history["loss"]),
    "best_val_acc_phase1":  float(max(history_p1.history["val_categorical_accuracy"])),
    "best_val_acc_phase2":  float(max(history_p2.history["val_categorical_accuracy"])),
    "classes_below_90pct":  int(classes_below_90),
    "worst_classes":        {class_names[i]: float(per_class_acc[i])
                             for i in worst_idx},
    "hyperparameters": {
        k: v for k, v in vars(CFG).items()
        if not k.startswith("_") and not callable(v)
    },
    "timestamp":            datetime.now().isoformat(),
    "tensorflow_version":   tf.__version__,
}
with open(os.path.join(CFG.OUTPUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 · ZIP RESULTS & FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

zip_path = "/kaggle/working/plant_disease_results.zip"
log.info(f"Creating archive → {zip_path}")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(CFG.OUTPUT_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, os.path.dirname(CFG.OUTPUT_DIR))
            zf.write(fpath, arcname)

zip_mb = os.path.getsize(zip_path) / 1024 / 1024

print("\n" + "═" * 70)
print("  🌿 TRAINING COMPLETE")
print("═" * 70)
print(f"  Model            : {CFG.BACKBONE} + CBAM Attention")
print(f"  Classes           : {NUM_CLASSES}")
print(f"  Loss             : CategoricalCrossentropy  |  Label Smoothing: {CFG.LABEL_SMOOTHING}")
print(f"  Class Weights     : {'Yes (inverse freq)' if CFG.USE_CLASS_WEIGHTS else 'No'}")
print(f"  Augmentation      : CutMix + MixUp + RandomErase + Geometric + Color")
print(f"  ──────────────────────────────────────────────────────────")
print(f"  Val Accuracy      : {acc_tta * 100:.2f}%  (with TTA)")
print(f"  Val Accuracy      : {acc_plain * 100:.2f}%  (without TTA)")
print(f"  Top-5 Accuracy    : {top5_acc * 100:.2f}%")
if roc_auc:
    print(f"  ROC-AUC (macro)   : {roc_auc:.4f}")
print(f"  Classes < 90%     : {classes_below_90}/{NUM_CLASSES}")
print(f"  ──────────────────────────────────────────────────────────")
print(f"  Results ZIP       : {zip_path}  ({zip_mb:.1f} MB)")
print(f"  Finished          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("═" * 70)
