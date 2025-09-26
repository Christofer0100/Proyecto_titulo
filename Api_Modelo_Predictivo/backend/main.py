from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

# ---------------------------
# Cargar modelos y encoders
# ---------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

try:
    modelo_origen = joblib.load(MODEL_DIR / "modelo_origen_rf.pkl")
    le_origen_target = joblib.load(MODEL_DIR / "label_encoder_origen.pkl")
except Exception as e:
    raise RuntimeError(f"Error cargando modelo de ORIGEN: {e}")

try:
    modelo_destino = joblib.load(MODEL_DIR / "modelo_destino_rf.pkl")
    le_origen_feature = joblib.load(MODEL_DIR / "label_encoder_origen_feature.pkl")
    le_destino_target = joblib.load(MODEL_DIR / "label_encoder_destino.pkl")
except Exception as e:
    raise RuntimeError(f"Error cargando modelo de DESTINO: {e}")

# ---------------------------
# App & CORS
# ---------------------------
app = FastAPI(title="Modelo Predictivo ATP Chile Open")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en local, permite todo. En producción se restringe.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Esquemas de entrada
# ---------------------------
class InputOrigen(BaseModel):
    dia_semana: int = Field(..., ge=0, le=6, description="0=Lun ... 6=Dom")
    hora_num: int = Field(..., ge=0, le=23, description="0..23")
    top_n: int = Field(3, ge=1, le=10)

class InputDestino(BaseModel):
    dia_semana: int = Field(..., ge=0, le=6)
    hora_num: int = Field(..., ge=0, le=23)
    origen: str
    top_n: int = Field(3, ge=1, le=10)

# ---------------------------
# Función auxiliar
# ---------------------------
def top_n_from_proba(probs: np.ndarray, enc, n: int, key_name: str):
    idx = probs.argsort()[::-1][:n]
    return [{key_name: enc.inverse_transform([i])[0], "prob": float(probs[i])} for i in idx]

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/meta")
def meta():
    return {
        "origenes_disponibles_modelo_origen": list(le_origen_target.classes_),
        "origenes_validos_para_destino": list(le_origen_feature.classes_),
        "destinos_disponibles": list(le_destino_target.classes_),
    }

@app.post("/predict_origen")
def predict_origen(inp: InputOrigen):
    Xq = np.array([[inp.hora_num, inp.dia_semana]])
    try:
        probs = modelo_origen.predict_proba(Xq)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir origen: {e}")
    return {"predicciones": top_n_from_proba(probs, le_origen_target, inp.top_n, "origen")}
    

@app.post("/predict_destino")
def predict_destino(inp: InputDestino):
    if inp.origen not in set(le_origen_feature.classes_):
        raise HTTPException(status_code=400, detail=f"Origen '{inp.origen}' no reconocido por el modelo de destino.")
    origen_enc = le_origen_feature.transform([inp.origen])[0]
    Xq = np.array([[inp.hora_num, inp.dia_semana, origen_enc]])
    try:
        probs = modelo_destino.predict_proba(Xq)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir destino: {e}")
    return {"predicciones": top_n_from_proba(probs, le_destino_target, inp.top_n, "destino")}

import json

@app.get("/metrics")
def metrics():
    payload = {
        "trained_at": None,
        "origen": {"accuracy": None, "f1_macro": None},
        "destino": {"accuracy": None, "f1_macro": None},
        "num_clases": {
            "origen": len(le_origen_target.classes_),
            "destino": len(le_destino_target.classes_)
        }
    }
    try:
        with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as f:
            file_metrics = json.load(f)
        payload.update(file_metrics)
    except Exception:
        # si no existe metrics.json, igual devolvemos el conteo de clases
        pass
    return payload

 