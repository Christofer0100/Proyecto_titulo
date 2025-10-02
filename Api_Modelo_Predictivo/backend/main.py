# main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

ORIGEN_MODEL_PATH  = MODEL_DIR / "model_origen.pkl"
DESTINO_MODEL_PATH = MODEL_DIR / "model_destino.pkl"
META_PATH          = MODEL_DIR / "metadata.json"

app = FastAPI(title="API Modelo Predictivo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Carga segura ----------
def _safe_load(path: Path):
    if not path.exists():
        return None
    try:
        return load(path)
    except Exception:
        return None

PIPE_ORIGEN  = _safe_load(ORIGEN_MODEL_PATH)
PIPE_DESTINO = _safe_load(DESTINO_MODEL_PATH)

# ---------- Helpers ----------
def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Rows -> DataFrame con columnas dia_semana y hora_num en int."""
    df = pd.DataFrame(rows or [])
    # normaliza tipos
    df["dia_semana"] = pd.to_numeric(df.get("dia_semana", 0), errors="coerce").fillna(0).astype(int)
    df["hora_num"]   = pd.to_numeric(df.get("hora_num", 0), errors="coerce").fillna(0).astype(int)
    return df[["dia_semana", "hora_num"]]

def _predict_with_proba(pipe, X: pd.DataFrame, top_n: int = 1):
    """
    Devuelve:
      - yhat: labels top1 (lista)
      - topk: lista de listas [{label, probability} ...] por fila
    """
    if pipe is None or X.empty:
        return [], [[]]

    # Predicción top1
    yhat = pipe.predict(X)

    # Probabilidades
    topk_all = [[] for _ in range(len(X))]
    try:
        proba = pipe.predict_proba(X)  # shape (n, n_classes)
        # clases en el estimador final del pipeline
        clf = pipe.named_steps.get("clf", None)
        classes = getattr(clf, "classes_", None) if clf is not None else None
        if classes is None:
            # fallback: intenta en el propio pipe (algunos wrappers exponen classes_)
            classes = getattr(pipe, "classes_", None)

        if classes is not None:
            import numpy as np
            for i, row in enumerate(proba):
                # índices de clases ordenados por prob desc
                idxs = np.argsort(row)[::-1][:max(1, top_n)]
                topk_all[i] = [
                    {"label": str(classes[j]), "probability": float(row[j])}
                    for j in idxs
                ]
    except Exception:
        # Si el estimador no soporta proba, dejamos topk vacío
        pass

    return list(map(str, yhat)), topk_all

def _format_response(task: str, labels: List[str], topk_row: List[Dict[str, Any]]):
    resp = {
        "task": task,
        "prediccion": [labels[0]] if labels else [],
    }
    if topk_row:
        resp["top1"] = {
            "label": topk_row[0]["label"],
            "probability": topk_row[0].get("probability")
        }
        resp["topk"] = topk_row
    return resp

# ---------- Health ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_origen_loaded": PIPE_ORIGEN is not None,
        "model_destino_loaded": PIPE_DESTINO is not None,
        "metadata_found": META_PATH.exists(),
    }

# ---------- Endpoints ----------
@app.post("/predict/origen")
def predict_origen(
    payload: Dict[str, Any] = Body(..., example={"rows":[{"dia_semana":2,"hora_num":10}],"top_n":1})
):
    rows  = payload.get("rows", [])
    top_n = int(payload.get("top_n", 1))
    X = _rows_to_df(rows)
    yhat, topk = _predict_with_proba(PIPE_ORIGEN, X, top_n=top_n)

    if not yhat:
        return {"task": "origen", "prediccion": []}

    return _format_response("origen", yhat, topk[0])

@app.post("/predict/destino")
def predict_destino(
    payload: Dict[str, Any] = Body(..., example={"rows":[{"dia_semana":2,"hora_num":10}],"top_n":1})
):
    rows  = payload.get("rows", [])
    top_n = int(payload.get("top_n", 1))
    X = _rows_to_df(rows)
    yhat, topk = _predict_with_proba(PIPE_DESTINO, X, top_n=top_n)

    if not yhat:
        return {"task": "destino", "prediccion": []}

    return _format_response("destino", yhat, topk[0])
