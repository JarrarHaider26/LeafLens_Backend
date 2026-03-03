"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🌿 LeafLens AI — FastAPI Backend Server                                    ║
║                                                                              ║
║  Serves predictions from the trained EfficientNetV2S + CBAM model.          ║
║  Designed to be called by the Next.js API route (/api/predict).             ║
║                                                                              ║
║  Endpoints:                                                                  ║
║    GET  /health   — Health check & model status                             ║
║    POST /predict  — Image classification with optional TTA                  ║
║                                                                              ║
║  Usage:                                                                      ║
║    python app.py                                                             ║
║    # or: uvicorn app:app --host 0.0.0.0 --port 8000                        ║
║                                                                              ║
║  Environment Variables:                                                      ║
║    MODEL_PATH  — Path to best_phase2.keras (default: ../Model/best_phase2.keras)
║    PORT        — Server port (default: 8000)                                ║
║    HOST        — Server host (default: 0.0.0.0)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("leaflens.server")

# ── Model Service (global singleton) ─────────────────────────────────────────
_model_service = None


def get_model_service():
    """Get the loaded model service instance."""
    global _model_service
    if _model_service is None:
        raise RuntimeError("Model not loaded")
    return _model_service


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model_service

    # Import here to avoid TF import before server starts
    from inference import ModelService, CLASS_NAMES

    # Resolve model path
    model_path = os.environ.get(
        "MODEL_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Model", "best_phase2.keras"),
    )
    model_path = os.path.normpath(model_path)

    logger.info("=" * 65)
    logger.info("  LeafLens AI Backend — Starting Up")
    logger.info(f"  Model: {model_path}")
    logger.info("=" * 65)

    try:
        _model_service = ModelService(model_path)
        _model_service.warmup()
        logger.info("✓ Server ready — accepting predictions")
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        logger.error("  Place your trained model at: Model/best_phase2.keras")
        raise
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down LeafLens backend...")
    _model_service = None


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="LeafLens AI Backend",
    description="Plant disease detection API powered by EfficientNetV2S + CBAM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js frontend (the Next.js API route calls this server-side,
# but we enable CORS for direct access during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.environ.get("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """Prediction request with base64-encoded image."""
    image: str = Field(..., description="Base64-encoded image (with or without data URL prefix)")
    tta: bool = Field(True, description="Enable Test-Time Augmentation for higher accuracy")
    tta_steps: int = Field(3, ge=1, le=10, description="Number of TTA augmentation views (default: 3)")
    top_k: int = Field(5, ge=1, le=38, description="Number of top predictions to return")


class PredictionItem(BaseModel):
    """Single class prediction."""
    class_name: str = Field(..., description="Raw PlantVillage class name")
    plant: str = Field(..., description="Plant species name")
    condition: str = Field(..., description="Disease or healthy condition")
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    class_index: int = Field(..., description="Class index (0-37)")


class PredictResponse(BaseModel):
    """Prediction response with top-k results and metadata."""
    predictions: list[PredictionItem]
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")
    tta_enabled: bool
    tta_views: int = Field(..., description="Total views used (1 = no TTA, 1+N = with TTA)")
    model: str = "EfficientNetV2S-CBAM"
    img_size: int = 300


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model_service is not None,
        "model": "EfficientNetV2S-CBAM",
        "classes": 38,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict plant disease from an image.
    
    The image is preprocessed exactly as during training:
      1. Resize to 300×300
      2. Normalize to [-1, 1] via (x / 127.5) - 1.0
    
    With TTA enabled (default), multiple augmented views are averaged
    for higher accuracy (matches the evaluation pipeline).
    """
    service = get_model_service()

    try:
        # ── Decode base64 image ───────────────────────────────────────────
        image_data = request.image
        if "," in image_data:
            # Strip data URL prefix: "data:image/jpeg;base64,..."
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        if len(image_bytes) < 100:
            raise HTTPException(status_code=400, detail="Image data too small — is it a valid image?")

        # ── Run inference ─────────────────────────────────────────────────
        start_time = time.time()

        if request.tta:
            probabilities = service.predict_with_tta(image_bytes, tta_steps=request.tta_steps)
        else:
            probabilities = service.predict(image_bytes)

        inference_ms = (time.time() - start_time) * 1000

        # ── Build top-k results ────────────────────────────────────────────
        from inference import CLASS_NAMES, ModelService as MS
        top_indices = probabilities.argsort()[::-1][: request.top_k]

        predictions = []
        for idx in top_indices:
            class_name = CLASS_NAMES[idx]
            plant, condition = MS.parse_class_name(class_name)
            predictions.append(
                PredictionItem(
                    class_name=class_name,
                    plant=plant,
                    condition=condition,
                    confidence=round(float(probabilities[idx]) * 100, 2),
                    class_index=int(idx),
                )
            )

        return PredictResponse(
            predictions=predictions,
            inference_time_ms=round(inference_ms, 1),
            tta_enabled=request.tta,
            tta_views=(request.tta_steps + 1) if request.tta else 1,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting LeafLens backend on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False, log_level="info")
