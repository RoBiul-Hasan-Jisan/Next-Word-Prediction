
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lstm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "transformer"))

app = FastAPI(title="Next Word Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Lazily-loaded predictors so the server starts instantly even if one of
# the two models isn't ready / isn't trained yet.
_lstm_predictor = None
_transformer_predictor = None


def get_lstm_predictor():
    global _lstm_predictor
    if _lstm_predictor is None:
        from predict_lstm import LSTMPredictor

        _lstm_predictor = LSTMPredictor()
    return _lstm_predictor


def get_transformer_predictor():
    global _transformer_predictor
    if _transformer_predictor is None:
        from predict_transformer import TransformerPredictor

        _transformer_predictor = TransformerPredictor()
    return _transformer_predictor


class PredictRequest(BaseModel):
    text: str
    engine: str = "transformer"  # "lstm" or "transformer"
    top_k: int = 5


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/predict")
async def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        return {"suggestions": []}

    try:
        if req.engine == "lstm":
            predictor = get_lstm_predictor()
        else:
            predictor = get_transformer_predictor()
        suggestions = predictor.predict(text, top_k=req.top_k)
    except FileNotFoundError as e:
        return {"suggestions": [], "error": str(e)}
    except Exception as e:  # pragma: no cover - surfaced to the UI
        return {"suggestions": [], "error": f"{type(e).__name__}: {e}"}

    return {"suggestions": suggestions}


@app.get("/health")
async def health():
    return {"status": "ok"}
