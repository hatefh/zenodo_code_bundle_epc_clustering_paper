import pandas as pd
import yaml
from pathlib import Path
from code.utils.paths import pth

# Optional scalers (imported lazily so script works even if user forgets to install something)
def get_scaler(kind: str):
    kind = (kind or "standard").lower()
    if kind == "raw":
        return None
    elif kind == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    elif kind == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    else:
        raise ValueError(f"Unknown scaling: {kind} (use raw|minmax|standard)")

def load_cfg():
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)

def _norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "_").replace("/", "_")

def find_col(df: pd.DataFrame, candidates):
    """
    Find a column in df whose name matches one of the candidates (case-insensitive, normalized).
    First tries exact normalized match, then substring match.
    """
    cols_norm = {_norm(c): c for c in df.columns}
    # exact normalized match
    for cand in candidates:
        key = _norm(cand)
        if key in cols_norm:
            return cols_norm[key]
    # substring fallback
    for c in df.columns:
        nc = _norm(c)
        for cand in candidates:
            if _norm(cand) in nc:
                return c
    raise KeyError(f"None of {candidates} found in EPC columns: {list(df.columns)}")

def to_binary(series: pd.Series, length: int) -> pd.Series:
    """
    Map common encodings of heat recovery to 0/1.
    If series is None (column missing), returns zeros.
    """
    if series is None:
        return pd.Series([0] * length, dtype=int)
    mapping = {
        "with_hr": 1, "with heat recovery": 1, "with": 1, "yes": 1, "y": 1, True: 1, 1: 1,
        "without_hr": 0, "without heat recovery": 0, "without": 0, "no": 0, "n": 0, False: 0, 0: 0
    }
    def m(x):
        if isinstance(x, str):
            return mapping.get(x.strip().lower(), None)
        return mapping.get(x, None)
    out = series.map(m)
    # try numeric fallback
    if out.isna().any():
        try:
            coerced = pd.to_numeric(series, errors="coerce").clip(lower=0, upper=1).astype("Int64")
            out = out.fillna(coerced)
        except Exception:
            pass
    return out.fillna(0).astype(int)

def main():
    cfg = load_cfg()
    scaling_kind = cfg.get("scaling", "standard")

    # Load cleaned EPC table produced by preprocessing
    epc_path = pth(cfg["outputs"]["epc_clean"])
    epc = pd.read_parquet(epc_path)
    n_rows = len(epc)

    # ---- Robust column mapping (works for both your schemas) ----
    # ID: old had 'building_id', new has 'no.' (or 'no')
    col_id          = find_col(epc, ["building_id", "id", "bldg_id", "no.", "no"])
    # EPC metrics
    col_ep_value    = find_col(epc, ["ep_value", "ep-value", "ep", "energy_performance"])
    col_weighted_u  = find_col(epc, ["weighted_u_value", "weighted u", "u_weighted", "u (weighted)", "u-weighted"])
    col_air_leakage = find_col(epc, ["air_leakage", "air_leakage_rate_m³/(h)", "q50", "q50_m3h", "airtightness"])
    col_dh_epc      = find_col(epc, ["dh_epc", "dh_kwh/m2a", "dh_kwh_m2", "annual_dh_per_m2", "dh (kwh/m2a)"])

    # Heat recovery (may be missing in 26-school file)
    try:
        col_hr = find_col(epc, ["ventilation_hr", "heat_recovery", "hr", "ventilation system", "ventilation_heat_recovery"])
        hr_series = epc[col_hr]
    except KeyError:
        hr_series = None

    # ---- Build feature frame ----
    feats = pd.DataFrame({
        "building_id": epc[col_id].values,
        "ep_value": pd.to_numeric(epc[col_ep_value], errors="coerce"),
        "weighted_u_value": pd.to_numeric(epc[col_weighted_u], errors="coerce"),
        "air_leakage": pd.to_numeric(epc[col_air_leakage], errors="coerce"),
        "dh_epc": pd.to_numeric(epc[col_dh_epc], errors="coerce"),
        "heat_recovery": to_binary(hr_series, n_rows),
    })

    # Impute any missing numeric values with median
    num_cols = ["ep_value", "weighted_u_value", "air_leakage", "dh_epc", "heat_recovery"]
    for c in num_cols:
        if feats[c].isna().any():
            feats[c] = feats[c].fillna(feats[c].median())

    # ---- Apply selected scaling ----
    scaler = get_scaler(scaling_kind)
    if scaler is None:  # raw
        feats_scaled = feats.copy()
    else:
        Xs = scaler.fit_transform(feats[num_cols])
        feats_scaled = pd.DataFrame(Xs, columns=num_cols, index=feats.index)
        feats_scaled.insert(0, "building_id", feats["building_id"].values)

    # ---- Save ----
    out_features = pth(cfg["outputs"]["features"])
    Path(out_features).parent.mkdir(parents=True, exist_ok=True)
    feats_scaled.to_parquet(out_features, index=False)
    print(f"EPC features extracted with scaling='{scaling_kind}' and saved to:", out_features)

if __name__ == "__main__":
    main()
