# train_and_export_calibrated.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from joblib import dump


# =========================
# Configuración de rutas
# =========================
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "ReservasAtp.csv"      # deja el CSV en /backend
OUT_DIR  = HERE / "model"                # carpeta de salida
OUT_DIR.mkdir(exist_ok=True)

ORIGEN_MODEL_PATH   = OUT_DIR / "model_origen.pkl"
DESTINO_MODEL_PATH  = OUT_DIR / "model_destino.pkl"
META_PATH           = OUT_DIR / "metadata.json"
METRICS_PATH        = OUT_DIR / "metrics.json"


# ======================================================
# Utilidades: normalización de nombres y detección cols
# ======================================================

def _norm(s: str) -> str:
    import unicodedata
    s = s.strip().lower()
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    s = s.replace(" ", "_").replace("-", "_").replace("/", "_")
    return s


def _infer_separator(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        head = ''.join([next(f) for _ in range(10)])
    return ';' if head.count(';') > head.count(',') else ','


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {_norm(c): c for c in df.columns}
    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    return {
        "datetime": pick([
            "fecha_hora", "fecha", "datetime", "fecha_salida", "salida", "created_at",
            "f_salida", "fecha_hora_salida", "hora_salida_completa"
        ]),
        "hour": pick(["hora_salida", "hora", "hora_inicio", "hora_reserva"]),
        "origen": pick(["origen", "origen_nombre", "lugar_origen"]),
        "destino": pick(["destino", "destino_nombre", "lugar_destino"]),
    }


# ============================================
# Construcción de features (día/hora) y target
# ============================================

def build_features(
    df: pd.DataFrame,
    col_map: Dict[str, Optional[str]],
    kind: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    if kind not in ("origen", "destino"):
        raise ValueError("kind debe ser 'origen' o 'destino'")

    data = df.copy()

    # fecha
    if col_map["datetime"] is not None:
        data["_dt"] = pd.to_datetime(
            data[col_map["datetime"]],
            errors="coerce", dayfirst=True
        )
    else:
        data["_dt"] = pd.NaT

    # hora
    if col_map["hour"] is not None:
        try:
            data["_hrtmp"] = pd.to_datetime(
                data[col_map["hour"]].astype(str),
                format="%H:%M", errors="coerce"
            )
            mask_na = data["_hrtmp"].isna()
            if mask_na.any():
                data.loc[mask_na, "_hrtmp"] = pd.to_datetime(
                    data.loc[mask_na, col_map["hour"]].astype(str),
                    format="%H:%M:%S", errors="coerce"
                )
        except Exception:
            data["_hrtmp"] = pd.NaT
    else:
        data["_hrtmp"] = pd.NaT

    # features
    data["hora_num"]   = np.where(
        data["_hrtmp"].notna(),
        data["_hrtmp"].dt.hour,
        data["_dt"].dt.hour
    )
    data["dia_semana"] = data["_dt"].dt.dayofweek
    data["dia_semana"] = data["dia_semana"].fillna(0).astype(int)
    data["hora_num"]   = data["hora_num"].fillna(0).astype(int)

    # target
    target_col = col_map[kind]
    if target_col is None:
        raise ValueError(f"No se encontró columna para '{kind}' en el CSV.")
    data[target_col] = data[target_col].astype(str).str.strip()
    data = data[(data[target_col].notna()) & (data[target_col] != "")]

    X = data[["dia_semana", "hora_num"]].copy()
    y = data[target_col].copy()

    return X, y


# =========================
# Entrenamiento calibrado
# =========================

def make_pipeline_calibrated() -> Pipeline:
    """
    Pipeline:
      - Preprocesador (numérico passthrough)
      - RandomForest
      - CalibratedClassifierCV (isotónica, cv=5)
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["dia_semana", "hora_num"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    base_rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight=None  # usa "balanced" si tu dataset está MUY desbalanceado
    )

    calibrated = CalibratedClassifierCV(
        estimator=base_rf,
        method="isotonic",
        cv=5,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", calibrated),
    ])
    return pipe


def fit_and_report(X: pd.DataFrame, y: pd.Series, etiqueta: str) -> Pipeline:
    """
    Entrena con calibración (cv=5) y calcula métricas
    por validación cruzada interna (aprox) usando predicciones in-sample.
    """
    if len(np.unique(y)) < 2:
        raise ValueError(f"'{etiqueta}': se necesitan al menos 2 clases para entrenar.")

    pipe = make_pipeline_calibrated()

    # Ajuste con calibración (cv=5 internamente)
    pipe.fit(X, y)

    # Predicción labels
    y_hat = pipe.predict(X)

    # Probabilidad de la clase predicha (para Brier tomamos prob de la clase verdadera con OVR)
    # Para Brier por multiclase usamos promedio de one-vs-rest
    proba = pipe.predict_proba(X)  # shape (n, n_classes)
    classes = pipe.named_steps["clf"].classes_
    classes = list(map(str, classes))
    y_true_idx = np.array([classes.index(str(t)) for t in y])
    p_true = proba[np.arange(len(y)), y_true_idx]

    acc = accuracy_score(y, y_hat)
    f1m = f1_score(y, y_hat, average="macro")
    brier = brier_score_loss((y == y).astype(int), p_true)  # trivial 0/1 para fórmula; usamos p_true directo abajo
    # corrección: brier_score_loss requiere y_true binaria; para multiclase usamos promedio OVR:
    # Implementación simple del Brier OVR
    y_bin = np.zeros_like(proba)
    for i, cls in enumerate(classes):
        y_bin[:, i] = (y.astype(str).values == cls).astype(int)
    brier_ovr = float(np.mean(np.mean((proba - y_bin) ** 2, axis=1)))

    print(f"\n=== {etiqueta.upper()} (calibrado isotónico, cv=5) ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1-macro:   {f1m:.4f}")
    print(f"Brier OVR:  {brier_ovr:.4f}")

    # Guardamos métricas para dashboard
    _save_partial_metrics(etiqueta, acc, f1m, brier_ovr)

    return pipe


# =============
# Guardado
# =============

def _save_partial_metrics(kind: str, acc: float, f1m: float, brier_ovr: float):
    METRICS_PATH.parent.mkdir(exist_ok=True)
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    else:
        metrics = {}

    metrics[kind] = {
        "accuracy": acc,
        "f1_macro": f1m,
        "brier_ovr": brier_ovr
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_save_model(pipe: Pipeline, out_path: Path) -> None:
    dump(pipe, out_path)
    print(f"✅ Modelo guardado en: {out_path.name}")


# ===============
# Programa princ.
# ===============

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV en {CSV_PATH}.\n"
            f"Asegúrate de colocar 'ReservasAtp.csv' dentro de la carpeta backend."
        )

    sep = _infer_separator(CSV_PATH)
    print(f"➤ Cargando CSV con separador '{sep}' ...")

    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig",
        sep=sep,
        engine="python"
    )
    print(f"✔ CSV cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Detecta columnas
    col_map = detect_columns(df)
    print("➤ Columnas detectadas:", json.dumps(col_map, ensure_ascii=False, indent=2))

    # Entrenamiento ORIGEN
    modelo_origen = None
    try:
        Xo, yo = build_features(df, col_map, kind="origen")
        if Xo.empty or yo.empty:
            print("⚠ No hay datos válidos para ORIGEN. Se omite.")
        else:
            print(f"➤ Entrenando (ORIGEN) con {len(Xo)} filas ...")
            modelo_origen = fit_and_report(Xo, yo, etiqueta="origen")
            safe_save_model(modelo_origen, ORIGEN_MODEL_PATH)
    except ValueError as e:
        print(f"⚠ ORIGEN: {e}")

    # Entrenamiento DESTINO (si existe)
    modelo_destino = None
    try:
        if col_map.get("destino"):
            Xd, yd = build_features(df, col_map, kind="destino")
            if Xd.empty or yd.empty:
                print("⚠ No hay datos válidos para DESTINO. Se omite.")
            else:
                print(f"➤ Entrenando (DESTINO) con {len(Xd)} filas ...")
                modelo_destino = fit_and_report(Xd, yd, etiqueta="destino")
                safe_save_model(modelo_destino, DESTINO_MODEL_PATH)
        else:
            print("ℹ No se encontró columna de DESTINO; no se entrena ese modelo.")
    except ValueError as e:
        print(f"⚠ DESTINO: {e}")

    # Guarda metadatos
    meta = {
        "source_csv": str(CSV_PATH.name),
        "separator": sep,
        "columns_detected": col_map,
        "features": ["dia_semana (0=lun..6=dom)", "hora_num (0..23)"],
        "models": {
            "origen": ORIGEN_MODEL_PATH.name if modelo_origen is not None else None,
            "destino": DESTINO_MODEL_PATH.name if modelo_destino is not None else None,
        },
        "calibrated": {
            "method": "isotonic",
            "cv": 5,
            "base_estimator": "RandomForestClassifier(n_estimators=300, random_state=42)"
        }
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📝 Metadatos guardados en: {META_PATH.name}")

    print("🎉 Entrenamiento calibrado finalizado.")


if __name__ == "__main__":
    main()
