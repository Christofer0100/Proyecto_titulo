# verify_probabilities.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import argparse
import json
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "ReservasAtp.csv"
MODEL_DIR = HERE / "model"
MODEL_ORIGEN  = MODEL_DIR / "model_origen.pkl"
MODEL_DESTINO = MODEL_DIR / "model_destino.pkl"
META_PATH     = MODEL_DIR / "metadata.json"

# ------------------ utilidades ------------------
def _infer_separator(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        head = ''.join([next(f) for _ in range(10)])
    return ';' if head.count(';') > head.count(',') else ','

def _norm(s: str) -> str:
    import unicodedata
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.replace(" ", "_").replace("-", "_").replace("/", "_")

def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {_norm(c): c for c in df.columns}
    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols: return cols[c]
        return None
    return {
        "datetime": pick(["fecha_hora","fecha","datetime","fecha_salida","salida","created_at","f_salida","fecha_hora_salida","hora_salida_completa"]),
        "hour":     pick(["hora_salida","hora","hora_inicio","hora_reserva"]),
        "origen":   pick(["origen","origen_nombre","lugar_origen"]),
        "destino":  pick(["destino","destino_nombre","lugar_destino"]),
    }

def build_features(df: pd.DataFrame, col_map: Dict[str, Optional[str]]) -> pd.DataFrame:
    data = df.copy()
    if col_map["datetime"] is not None:
        data["_dt"] = pd.to_datetime(data[col_map["datetime"]], errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        data["_dt"] = pd.NaT
    if col_map["hour"] is not None:
        try:
            data["_hrtmp"] = pd.to_datetime(data[col_map["hour"]].astype(str), format="%H:%M", errors="coerce")
            mask = data["_hrtmp"].isna()
            if mask.any():
                data.loc[mask, "_hrtmp"] = pd.to_datetime(data.loc[mask, col_map["hour"]].astype(str), format="%H:%M:%S", errors="coerce")
        except Exception:
            data["_hrtmp"] = pd.NaT
    else:
        data["_hrtmp"] = pd.NaT

    data["hora_num"]   = np.where(data["_hrtmp"].notna(), data["_hrtmp"].dt.hour, data["_dt"].dt.hour)
    data["dia_semana"] = data["_dt"].dt.dayofweek
    data["dia_semana"] = data["dia_semana"].fillna(0).astype(int)
    data["hora_num"]   = data["hora_num"].fillna(0).astype(int)
    return data

def empirical_distribution(df: pd.DataFrame, target_col: str, day: int, hour: int) -> Tuple[Optional[str], float, pd.DataFrame]:
    """
    Devuelve (label_top, freq_top, tabla_frecuencias_filtrada)
    freq_top en [0..1]
    """
    sub = df[(df["dia_semana"]==day) & (df["hora_num"]==hour)]
    if sub.empty or target_col not in sub.columns:
        return None, 0.0, pd.DataFrame()
    # limpiar target
    t = sub[target_col].astype(str).str.strip()
    t = t[(t.notna()) & (t!="")]
    if t.empty:
        return None, 0.0, pd.DataFrame()
    counts = t.value_counts(dropna=False)
    total = counts.sum()
    freqs = (counts / total).rename("freq").to_frame()
    top_label = counts.idxmax()
    top_freq  = (counts.max() / total) if total>0 else 0.0
    # ordenar por freq desc para impresión
    freqs = freqs.sort_values("freq", ascending=False).reset_index().rename(columns={"index":target_col})
    return str(top_label), float(top_freq), freqs

def load_pipe(kind: str):
    path = MODEL_ORIGEN if kind=="origen" else MODEL_DESTINO
    return load(path) if path.exists() else None

def model_top1_and_proba(pipe, Xrow: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    """
    Predice una fila y devuelve (label, prob_top1) si el modelo soporta predict_proba.
    """
    if pipe is None or Xrow is None or Xrow.empty: return None, None
    yhat = pipe.predict(Xrow)[0]
    prob = None
    try:
        proba = pipe.predict_proba(Xrow)[0]  # (n_classes,)
        clf   = pipe.named_steps.get("clf", None)
        classes = getattr(clf, "classes_", None) if clf is not None else getattr(pipe, "classes_", None)
        if classes is not None:
            classes = list(map(str, classes))
            idx = classes.index(str(yhat))
            prob = float(proba[idx])
    except Exception:
        prob = None
    return str(yhat), prob

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(description="Comparar prob del modelo vs frecuencia empírica del CSV")
    ap.add_argument("--kind", choices=["origen","destino"], default="origen", help="Qué modelo verificar")
    ap.add_argument("--day", type=int, required=True, help="día 0=lun .. 6=dom")
    ap.add_argument("--hour", type=int, required=False, help="hora 0..23; si se omite, imprime tabla de 24h")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No existe {CSV_PATH}")

    sep = _infer_separator(CSV_PATH)
    df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig", sep=sep, engine="python")
    col_map = detect_columns(df_raw)
    target_col = col_map[args.kind]
    if target_col is None:
        raise RuntimeError(f"No se encontró columna target para {args.kind} en el CSV")

    df_feat = build_features(df_raw, col_map)

    # unir features al df original para poder filtrar por día/hora y usar el target original
    df_all = pd.concat([df_raw.reset_index(drop=True), df_feat[["dia_semana","hora_num"]].reset_index(drop=True)], axis=1)

    pipe = load_pipe(args.kind)
    if pipe is None:
        raise RuntimeError(f"No se encontró el modelo {args.kind} en {MODEL_DIR}")

    def check_one(day:int, hour:int):
        # modelo
        Xrow = pd.DataFrame({"dia_semana":[day], "hora_num":[hour]})
        m_label, m_prob = model_top1_and_proba(pipe, Xrow)

        # csv empírico
        e_label, e_freq, table = empirical_distribution(df_all, target_col, day, hour)

        print(f"\n>>> {args.kind.upper()} | día={day}, hora={hour:02d}:00")
        print(f"Modelo:     top1 = {m_label}  | prob_top1 = {None if m_prob is None else round(m_prob*100,2)}%")
        print(f"CSV empír.: top1 = {e_label}  | freq_top1 = {round(e_freq*100,2)}%  (n={table['freq'].sum():.0f} proporciones)")
        # Mostrar top 5 del CSV
        if not table.empty:
            print("\nTop CSV (hasta 5):")
            for _, row in table.head(5).iterrows():
                print(f" - {row[target_col]:<25} {row['freq']*100:5.2f}%")

    if args.hour is None:
        # tabla 24h
        print(f"=== {args.kind.upper()} | DÍA {args.day} (0=lun..6=dom) ===")
        for h in range(24):
            check_one(args.day, h)
    else:
        check_one(args.day, args.hour)

if __name__ == "__main__":
    main()
