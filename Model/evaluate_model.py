"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Plant Disease Detection — TTA Evaluation & Export Script                    ║
║  Loads best_phase2.keras, runs TTA + standard evaluation, exports results   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · CUSTOM LAYERS (must be defined BEFORE loading model)
# ═══════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="Custom", name="ChannelAttention")
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


@tf.keras.utils.register_keras_serializable(package="Custom", name="SpatialAttention")
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


@tf.keras.utils.register_keras_serializable(package="Custom", name="CBAM")
class CBAM(layers.Layer):
    """Convolutional Block Attention Module: channel -> spatial."""

    def __init__(self, channels, reduction=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channels   = channels
        self.reduction  = reduction
        self.kernel_size = kernel_size
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def call(self, x):
        return self.sa(self.ca(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "channels":    self.channels,
            "reduction":   self.reduction,
            "kernel_size": self.kernel_size,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="Custom", name="WarmupCosineSchedule")
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup -> Cosine decay to min_lr."""

    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=1e-7):
        super().__init__()
        self.base_lr      = base_lr
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr       = min_lr

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
# SECTION 2 · CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_ROOT = "/kaggle/input/new-plant-diseases-dataset"
DATASET_ROOTS_FALLBACK = [
    "/kaggle/input/new-plant-diseases-dataset",
    "/kaggle/input/datasets/vipoooool/new-plant-diseases-dataset",
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)",
    "/kaggle/input/new-plant-diseases-dataset/new plant diseases dataset(augmented)",
]

# ── Model path: try multiple known locations ──────────────────────────────────
MODEL_PATHS = [
    "/kaggle/input/models/muhammadjavedraza/jarrar/keras/default/1/best_phase2.keras",
    "/kaggle/input/jarrar/best_phase2.keras",
    "/kaggle/working/results/best_phase2.keras",
    "/kaggle/working/best_phase2.keras",
]

IMG_SIZE    = 300
BATCH_SIZE  = 32
TTA_STEPS   = 3
OUTPUT_DIR  = "/kaggle/working/results_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · DATASET & MODEL PATH DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def _try_find_splits(root):
    """Check if root (or known sub-dirs) contains train/ and valid/."""
    if not os.path.exists(root):
        return None, None
    for sub in [
        "",
        "New Plant Diseases Dataset(Augmented)",
        "new plant diseases dataset(augmented)",
        "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
    ]:
        path = os.path.join(root, sub) if sub else root
        train = os.path.join(path, "train")
        valid = os.path.join(path, "valid")
        if os.path.isdir(train) and os.path.isdir(valid):
            return train, valid
    return None, None


def find_data_dirs():
    """3-strategy dataset discovery (matches main training script)."""
    # 1. Try primary DATASET_ROOT
    train_dir, valid_dir = _try_find_splits(DATASET_ROOT)
    if train_dir and valid_dir:
        return train_dir, valid_dir
    # 2. Try each fallback
    for root in DATASET_ROOTS_FALLBACK:
        train_dir, valid_dir = _try_find_splits(root)
        if train_dir and valid_dir:
            return train_dir, valid_dir
    # 3. Recursively scan /kaggle/input/
    for root, dirs, _ in os.walk("/kaggle/input/"):
        if "train" in dirs and "valid" in dirs:
            return os.path.join(root, "train"), os.path.join(root, "valid")
    raise FileNotFoundError(
        "Could not find train/ and valid/ directories. "
        "Make sure the dataset is added to this notebook."
    )


def find_model_path():
    """Try multiple known model paths."""
    for p in MODEL_PATHS:
        if os.path.isfile(p):
            return p
    # Recursively scan /kaggle/input/ for best_phase2.keras
    for root, _, files in os.walk("/kaggle/input/"):
        for f in files:
            if f == "best_phase2.keras":
                return os.path.join(root, f)
    raise FileNotFoundError(
        "Could not find best_phase2.keras. "
        "Make sure the model is added to this notebook."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · LOAD DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  Discovering dataset and model paths ...")
print("=" * 60)

train_dir, valid_dir = find_data_dirs()
print(f"  Train dir : {train_dir}")
print(f"  Valid dir : {valid_dir}")

model_path = find_model_path()
print(f"  Model path: {model_path}")

# ── Class names ───────────────────────────────────────────────────────────────
class_names = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])
NUM_CLASSES = len(class_names)
print(f"  Classes   : {NUM_CLASSES}")

# ── Validation dataset ────────────────────────────────────────────────────────
# Raw dataset: images in [0, 255] — needed for TTA augmentation
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
total_val = sum(1 for _ in val_ds_raw.unbatch())
print(f"  Val images: {total_val}")

# Normalised dataset: images in [-1, 1] — matches training preprocessing
def preprocess_val(images, labels):
    """Normalise to [-1, 1] (EfficientNetV2 preprocessing)."""
    images = tf.cast(images, tf.float32)
    images = (images / 127.5) - 1.0
    return images, labels

val_ds = val_ds_raw.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

# ── Load model (compile=False skips optimizer deserialization) ─────────────────
custom_objects = {
    "CBAM":                CBAM,
    "ChannelAttention":    ChannelAttention,
    "SpatialAttention":    SpatialAttention,
    "WarmupCosineSchedule": WarmupCosineSchedule,
}

print("\n  Loading model (compile=False for inference) ...")
model = tf.keras.models.load_model(
    model_path,
    custom_objects=custom_objects,
    compile=False,
)
print("  Model loaded successfully!")
model.summary(print_fn=lambda x: print(f"  {x}"))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · STANDARD EVALUATION (without TTA)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Running standard evaluation (no TTA) ...")
print("=" * 60)

# Compile for evaluation metrics only
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
    ],
)

results = model.evaluate(val_ds, verbose=1)
print(f"\n  Standard Val Loss     : {results[0]:.4f}")
print(f"  Standard Val Accuracy : {results[1]*100:.2f}%")
print(f"  Standard Val Top-5    : {results[2]*100:.2f}%")

# Get predictions without TTA
y_true_list = []
y_pred_plain_list = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred_plain_list.extend(np.argmax(preds, axis=1))
    y_true_list.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true_list)
y_pred_plain = np.array(y_pred_plain_list)
acc_plain = np.mean(y_true == y_pred_plain)
print(f"  Plain Accuracy        : {acc_plain*100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · TTA EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"  Running TTA evaluation (steps={TTA_STEPS}) ...")
print("=" * 60)


def tta_predict(model, dataset_raw, tta_steps=3):
    """Predict with TTA. Input: raw [0,255] images. Normalises internally."""
    y_true = []
    y_pred_probs = []
    batch_count = 0

    for images, labels in dataset_raw:
        images = tf.cast(images, tf.float32)
        tta_preds = []

        # Original predictions (normalise to [-1, 1])
        norm_images = (images / 127.5) - 1.0
        preds_orig = model.predict(norm_images, verbose=0)
        tta_preds.append(preds_orig)

        # Augmented predictions
        for _ in range(tta_steps - 1):
            aug_images = tf.image.random_flip_left_right(images)
            aug_images = tf.image.random_brightness(aug_images, 0.1)
            aug_images = tf.image.random_contrast(aug_images, lower=0.9, upper=1.1)
            aug_images = tf.clip_by_value(aug_images, 0.0, 255.0)
            norm_aug = (aug_images / 127.5) - 1.0
            preds = model.predict(norm_aug, verbose=0)
            tta_preds.append(preds)

        avg_preds = np.mean(tta_preds, axis=0)
        y_pred_probs.extend(avg_preds)
        y_true.extend(np.argmax(labels.numpy(), axis=1))

        batch_count += 1
        if batch_count % 50 == 0:
            print(f"    Processed {batch_count} batches ...")

    return np.array(y_true), np.array(y_pred_probs)


y_true_tta, y_pred_probs_tta = tta_predict(model, val_ds_raw, tta_steps=TTA_STEPS)
y_pred_tta = np.argmax(y_pred_probs_tta, axis=1)
acc_tta = np.mean(y_true_tta == y_pred_tta)
print(f"\n  TTA Accuracy: {acc_tta*100:.2f}%")

# Top-5 accuracy with TTA
top5_preds = np.argsort(y_pred_probs_tta, axis=1)[:, -5:]
top5_correct = sum(1 for i, t in enumerate(y_true_tta) if t in top5_preds[i])
top5_acc = top5_correct / len(y_true_tta)
print(f"  TTA Top-5 Accuracy: {top5_acc*100:.2f}%")

# ROC-AUC (macro)
try:
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true_tta, classes=list(range(NUM_CLASSES)))
    roc_auc = roc_auc_score(y_true_bin, y_pred_probs_tta, multi_class="ovr", average="macro")
    print(f"  ROC-AUC (macro): {roc_auc:.4f}")
except Exception as e:
    roc_auc = None
    print(f"  ROC-AUC skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 · CONFUSION MATRIX & CLASSIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Generating reports and plots ...")
print("=" * 60)

cm = confusion_matrix(y_true_tta, y_pred_tta)

# Classification report (JSON)
report_dict = classification_report(
    y_true_tta, y_pred_tta, target_names=class_names, output_dict=True
)
report_str = classification_report(
    y_true_tta, y_pred_tta, target_names=class_names
)
print("\n" + report_str)

with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
    json.dump(report_dict, f, indent=2)

np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 · PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Confusion Matrix ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 18))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=16)
ax.set_xlabel("Predicted", fontsize=14)
ax.set_ylabel("True", fontsize=14)
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(class_names, rotation=90, fontsize=6)
ax.set_yticklabels(class_names, fontsize=6)
plt.colorbar(im)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ── Per-Class Accuracy ────────────────────────────────────────────────────────
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
fig.savefig(os.path.join(OUTPUT_DIR, "per_class_accuracy.png"), dpi=150)
plt.close()

# ── Worst-performing classes ──────────────────────────────────────────────────
worst_idx = np.argsort(per_class_acc)[:5]
print("\n  5 worst-performing classes:")
for i in worst_idx:
    print(f"    [{i:2d}] {class_names[i]}: {per_class_acc[i]*100:.1f}%")

classes_below_90 = sum(1 for a in per_class_acc if a < 0.90)
print(f"\n  Classes below 90%: {classes_below_90}/{NUM_CLASSES}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 · CLASS MAPPING & SUMMARY JSON
# ═══════════════════════════════════════════════════════════════════════════════

# ── Class mapping JSON ────────────────────────────────────────────────────────
class_mapping = {str(i): name for i, name in enumerate(class_names)}
with open(os.path.join(OUTPUT_DIR, "class_mapping.json"), "w") as f:
    json.dump(class_mapping, f, indent=2)

# ── Training summary JSON ────────────────────────────────────────────────────
from datetime import datetime

summary = {
    "num_classes":          NUM_CLASSES,
    "class_names":          class_names,
    "total_val_samples":    int(total_val),
    "accuracy_with_tta":    float(acc_tta),
    "accuracy_without_tta": float(acc_plain),
    "top5_accuracy":        float(top5_acc),
    "roc_auc_macro":        float(roc_auc) if roc_auc else None,
    "classes_below_90pct":  int(classes_below_90),
    "worst_classes":        {class_names[i]: float(per_class_acc[i]) for i in worst_idx},
    "tta_steps":            TTA_STEPS,
    "image_size":           IMG_SIZE,
    "model_path":           model_path,
    "timestamp":            datetime.now().isoformat(),
    "tensorflow_version":   tf.__version__,
}
with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 · FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  EVALUATION COMPLETE")
print("=" * 70)
print(f"  Classes           : {NUM_CLASSES}")
print(f"  Val Accuracy      : {acc_tta * 100:.2f}%  (with TTA)")
print(f"  Val Accuracy      : {acc_plain * 100:.2f}%  (without TTA)")
print(f"  Top-5 Accuracy    : {top5_acc * 100:.2f}%")
if roc_auc:
    print(f"  ROC-AUC (macro)   : {roc_auc:.4f}")
print(f"  Classes < 90%     : {classes_below_90}/{NUM_CLASSES}")
print(f"  Results saved to  : {OUTPUT_DIR}")
print(f"  Finished          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
