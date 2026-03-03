"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🌿 LeafLens AI — Model Inference Service                                   ║
║                                                                              ║
║  Serves the trained EfficientNetV2S + CBAM plant disease detection model.   ║
║  ALL preprocessing steps are exact matches of the training pipeline          ║
║  (plant_disease_kaggle_final.py) to ensure identical accuracy.              ║
║                                                                              ║
║  Architecture : EfficientNetV2S + CBAM Attention (Woo et al., ECCV 2018)   ║
║  Input Size   : 300 × 300 × 3                                               ║
║  Preprocessing: (x / 127.5) - 1.0  →  normalize to [-1, 1]                ║
║  TTA          : horizontal flip, rot90, brightness ±0.1, contrast 0.9–1.1  ║
║  Classes      : 38 PlantVillage disease classes                              ║
║                                                                              ║
║  IMPORTANT: Custom layers (ChannelAttention, SpatialAttention, CBAM) are    ║
║  exact copies from the training script. Any modification will break model   ║
║  loading. Do NOT modify the layer implementations.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import io
import os
import time
import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger("leaflens.inference")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Must match training script exactly
# ═══════════════════════════════════════════════════════════════════════════════

IMG_SIZE = 300          # EfficientNetV2S input size (matches CFG.IMG_SIZE)
NUM_CLASSES = 38        # PlantVillage dataset classes
TTA_STEPS_DEFAULT = 3   # Default TTA augmentation views (matches CFG.TTA_STEPS)

# 38 PlantVillage classes — alphabetically sorted by directory name.
# This order MUST match `tf.keras.utils.image_dataset_from_directory` output
# used during training. Python's `sorted()` on directory names produces this
# exact ordering (case-sensitive ASCII sort on Linux/Kaggle).
CLASS_NAMES = [
    "Apple___Apple_scab",                                      # 0
    "Apple___Black_rot",                                       # 1
    "Apple___Cedar_apple_rust",                                # 2
    "Apple___healthy",                                         # 3
    "Blueberry___healthy",                                     # 4
    "Cherry_(including_sour)___Powdery_mildew",                # 5
    "Cherry_(including_sour)___healthy",                       # 6
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",      # 7
    "Corn_(maize)___Common_rust_",                             # 8
    "Corn_(maize)___Northern_Leaf_Blight",                     # 9
    "Corn_(maize)___healthy",                                  # 10
    "Grape___Black_rot",                                       # 11
    "Grape___Esca_(Black_Measles)",                            # 12
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",              # 13
    "Grape___healthy",                                         # 14
    "Orange___Haunglongbing_(Citrus_greening)",                # 15
    "Peach___Bacterial_spot",                                  # 16
    "Peach___healthy",                                         # 17
    "Pepper,_bell___Bacterial_spot",                           # 18
    "Pepper,_bell___healthy",                                  # 19
    "Potato___Early_blight",                                   # 20
    "Potato___Late_blight",                                    # 21
    "Potato___healthy",                                        # 22
    "Raspberry___healthy",                                     # 23
    "Soybean___healthy",                                       # 24
    "Squash___Powdery_mildew",                                 # 25
    "Strawberry___Leaf_scorch",                                # 26
    "Strawberry___healthy",                                    # 27
    "Tomato___Bacterial_spot",                                 # 28
    "Tomato___Early_blight",                                   # 29
    "Tomato___Late_blight",                                    # 30
    "Tomato___Leaf_Mold",                                      # 31
    "Tomato___Septoria_leaf_spot",                              # 32
    "Tomato___Spider_mites Two-spotted_spider_mite",           # 33
    "Tomato___Target_Spot",                                    # 34
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",                  # 35
    "Tomato___Tomato_mosaic_virus",                            # 36
    "Tomato___healthy",                                        # 37
]

assert len(CLASS_NAMES) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(CLASS_NAMES)}"


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM LAYERS — Exact copies from training script (plant_disease_kaggle_final.py)
#
# These MUST be registered before model loading. Both the decorator AND the
# custom_objects dict are used for maximum compatibility across TF versions.
#
# References:
#   - CBAM: Woo et al., "Convolutional Block Attention Module", ECCV 2018
#   - ChannelAttention: learn 'what' features to attend to (channel-wise)
#   - SpatialAttention: learn 'where' to attend (spatial-wise)
# ═══════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="PlantDisease")
class ChannelAttention(tf.keras.layers.Layer):
    """Learn 'what' features to attend to (channel-wise)."""

    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        mid = max(channels // reduction, 8)
        self.fc1 = tf.keras.layers.Dense(mid, activation="relu", use_bias=False)
        self.fc2 = tf.keras.layers.Dense(channels, use_bias=False)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        mx = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        att = tf.sigmoid(self.fc2(self.fc1(avg)) + self.fc2(self.fc1(mx)))
        return x * att

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "reduction": self.reduction})
        return cfg


@tf.keras.utils.register_keras_serializable(package="PlantDisease")
class SpatialAttention(tf.keras.layers.Layer):
    """Learn 'where' to attend (spatial-wise)."""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(
            1, kernel_size, padding="same", activation="sigmoid", use_bias=False
        )

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        return x * self.conv(tf.concat([avg, mx], axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg


@tf.keras.utils.register_keras_serializable(package="PlantDisease")
class CBAM(tf.keras.layers.Layer):
    """Convolutional Block Attention Module: channel → spatial."""

    def __init__(self, channels, reduction=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
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


# ═══════════════════════════════════════════════════════════════════════════════
# TTA AUGMENTATION — Exact copy from training script Section 15
#
# Operates on UNNORMALIZED images [0, 255] float32.
# Normalization is applied AFTER augmentation, matching training exactly.
# ═══════════════════════════════════════════════════════════════════════════════

def tta_augment_single(image: tf.Tensor) -> tf.Tensor:
    """
    One random augmentation for TTA (operates on 0-255 float image).
    
    Exact copy from training script `tta_single()`:
      - Random horizontal flip
      - Random 90° rotation (0, 90, 180, or 270 degrees)
      - Random brightness adjustment (±0.1)
      - Random contrast adjustment (0.9–1.1)
    """
    image = tf.image.random_flip_left_right(image)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class ModelService:
    """
    Handles model loading, preprocessing, inference, and TTA.
    
    Preprocessing pipeline (exact match with training):
      1. Load image as RGB
      2. Resize to 300×300 (bilinear interpolation)
      3. Cast to float32
      4. Normalize: (x / 127.5) - 1.0  →  range [-1, 1]
    
    TTA pipeline (exact match with training):
      1. Base prediction on normalized image
      2. N augmented views: augment raw [0,255] → normalize → predict
      3. Average all (1 + N) predictions
    """

    def __init__(self, model_path: str):
        self.model: Optional[tf.keras.Model] = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the trained Keras model with custom CBAM layers."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please place best_phase2.keras in the Model/ directory."
            )

        logger.info(f"Loading model from: {self.model_path}")
        start = time.time()

        # Custom objects dict for model deserialization
        custom_objects = {
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
            "CBAM": CBAM,
        }

        # compile=False skips optimizer deserialization (WarmupCosineSchedule)
        # which is not needed for inference and avoids compatibility issues.
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects=custom_objects,
            compile=False,
        )

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

        # Verify model output shape
        output_shape = self.model.output_shape
        if output_shape[-1] != NUM_CLASSES:
            raise ValueError(
                f"Model output shape {output_shape} does not match "
                f"expected {NUM_CLASSES} classes. Check CLASS_NAMES."
            )
        logger.info(f"Model output: {output_shape} ({NUM_CLASSES} classes) ✓")

    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess an image for inference.
        
        Returns:
            normalized: preprocessed image in [-1, 1] range, shape (300, 300, 3)
            raw: original resized image in [0, 255] range for TTA augmentation
        
        Preprocessing matches training `preprocess_val()`:
            image = tf.cast(image, tf.float32)
            image = (image / 127.5) - 1.0
        """
        # Load image with PIL (same backend as Keras image_dataset_from_directory)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        
        # Convert to float32 array [0, 255]
        raw = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1] — exact match with training preprocessing
        normalized = (raw / 127.5) - 1.0
        
        return normalized, raw

    def predict(self, image_bytes: bytes) -> np.ndarray:
        """
        Single prediction without TTA.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            Probability array of shape (38,) — one probability per class
        """
        normalized, _ = self.preprocess(image_bytes)
        batch = np.expand_dims(normalized, axis=0)  # (1, 300, 300, 3)
        preds = self.model.predict(batch, verbose=0)
        return preds[0]

    def predict_with_tta(self, image_bytes: bytes, tta_steps: int = TTA_STEPS_DEFAULT) -> np.ndarray:
        """
        Prediction with Test-Time Augmentation (TTA).
        
        Exact match with training script `predict_with_tta()`:
          1. Base prediction on normalized clean image
          2. For each TTA step: augment raw image → normalize → predict
          3. Average all predictions: accumulated / (1 + tta_steps)
        
        Optimization: All TTA views are batched into a single forward pass.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            tta_steps: Number of augmented views (default: 3, matching training)
            
        Returns:
            Averaged probability array of shape (38,)
        """
        normalized, raw = self.preprocess(image_bytes)
        
        # Collect all views: base + augmented
        views = [normalized]  # Start with clean normalized image
        
        raw_tensor = tf.constant(raw)  # [0, 255] for TTA augmentation
        
        for _ in range(tta_steps):
            # Augment on raw [0, 255] — matches training exactly
            augmented = tta_augment_single(raw_tensor).numpy()
            # Then normalize — matches training flow
            aug_normalized = (augmented / 127.5) - 1.0
            views.append(aug_normalized)
        
        # Batch all views for efficient single forward pass
        batch = np.stack(views, axis=0)  # (1 + tta_steps, 300, 300, 3)
        all_preds = self.model.predict(batch, verbose=0)  # (1 + tta_steps, 38)
        
        # Average predictions across all views
        avg_preds = np.mean(all_preds, axis=0)  # (38,)
        
        return avg_preds

    def warmup(self):
        """
        Run a dummy prediction to compile the TF graph.
        First call triggers graph tracing (~2-5s), subsequent calls are fast.
        """
        logger.info("Warming up model (compiling TF graph)...")
        dummy = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Convert to bytes via PIL
        img = Image.fromarray(dummy, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        dummy_bytes = buf.getvalue()
        
        # Run prediction to trigger graph compilation
        start = time.time()
        _ = self.predict_with_tta(dummy_bytes, tta_steps=1)
        elapsed = time.time() - start
        logger.info(f"Warmup complete ({elapsed:.1f}s)")

    @staticmethod
    def parse_class_name(class_name: str) -> tuple[str, str]:
        """
        Parse a PlantVillage class name into (plant, condition).
        
        Examples:
            "Apple___Apple_scab" → ("Apple", "Apple scab")
            "Cherry_(including_sour)___Powdery_mildew" → ("Cherry (including sour)", "Powdery mildew")
            "Tomato___healthy" → ("Tomato", "Healthy")
            "Corn_(maize)___Common_rust_" → ("Corn (maize)", "Common rust")
        """
        parts = class_name.split("___")
        
        # Clean plant name: replace underscores, handle parentheses
        plant = parts[0].replace("_", " ").replace(",", ", ").strip()
        # Clean double spaces
        plant = " ".join(plant.split())
        
        # Clean condition name
        if len(parts) > 1:
            condition = parts[1].replace("_", " ").strip()
            condition = " ".join(condition.split())
            # Capitalize first letter
            condition = condition[0].upper() + condition[1:] if condition else "Unknown"
        else:
            condition = "Unknown"
        
        return plant, condition
