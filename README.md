# LeafLens AI Backend

FastAPI backend for plant disease detection using EfficientNetV2S + CBAM model.

## 🚀 Quick Start (Local)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`

## 📁 Project Structure

```
leaflens-backend/
├── app.py              # FastAPI server
├── inference.py        # Model inference logic
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── Model/
│   └── best_phase2.keras   # Trained model (199MB)
└── README.md
```

## 🌐 API Endpoints

### Health Check
```
GET /health
```
Returns model status and version info.

### Predict Disease
```
POST /predict
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZ...",
  "tta": true,
  "tta_steps": 3,
  "top_k": 5
}
```

Response:
```json
{
  "predictions": [
    {
      "class_name": "Tomato___Early_blight",
      "plant": "Tomato",
      "condition": "Early blight",
      "confidence": 98.5,
      "class_index": 28
    }
  ],
  "inference_time_ms": 145.2,
  "tta_enabled": true,
  "tta_views": 3,
  "model": "EfficientNetV2S_CBAM"
}
```

## 🚀 Deploy to Render

### Option 1: Blueprint (Recommended)
1. Fork/push this repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Blueprint"
4. Connect your GitHub repo
5. Render will auto-detect `render.yaml`

### Option 2: Manual Setup
1. Create new "Web Service" on Render
2. Connect your GitHub repo
3. Configure:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add environment variable:
   - `MODEL_PATH`: `./Model/best_phase2.keras`

### ⚠️ Important Notes
- The model file (`best_phase2.keras`, ~199MB) must be in the repo
- Use Render's free tier for testing; upgrade for production (model loading needs ~2GB RAM)
- First request may be slow (cold start + model loading)

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./Model/best_phase2.keras` | Path to trained model |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |

## 📊 Model Info

- **Architecture**: EfficientNetV2S + CBAM Attention
- **Accuracy**: 99.98% on PlantVillage dataset
- **Classes**: 38 plant disease categories
- **Input Size**: 224x224 RGB images
- **Training Dataset**: 87,000+ images from PlantVillage

## 🔗 Frontend Integration

The frontend (Next.js) should set the `BACKEND_URL` environment variable:

```env
BACKEND_URL=https://your-backend.onrender.com
```

## 📝 License

MIT License
