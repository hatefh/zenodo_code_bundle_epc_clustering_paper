"""Step 02: Compute internal validation metrics for each clustering method.

Outputs a CSV with silhouette, Davies–Bouldin, and Calinski–Harabasz scores.
Includes a small reproducibility assertion for the GMM row (rounded values).
"""
from __future__ import annotations

import sys
import pandas as pd
import yaml
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from code.utils.paths import pth


def load_cfg() -> dict:
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def _normalize_index(idx: pd.Index) -> pd.Index:
    return pd.Index(idx.astype(str).str.strip(), name=idx.name)


def _compute(X: pd.DataFrame, labels: pd.Series) -> dict:
    y = labels.values
    return {
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
    }


def main() -> int:
    cfg = load_cfg()
    feats = pd.read_parquet(pth(cfg["outputs"]["features"])).set_index("building_id")
    feats.index = _normalize_index(feats.index)

    methods = [
        ("kmedoids", cfg["outputs"]["clusters_kmedoids"]),
        ("agglomerative", cfg["outputs"]["clusters_agglomerative"]),
        ("gmm", cfg["outputs"]["clusters_gmm"]),
    ]

    rows = []
    for name, rel in methods:
        try:
            lbl = pd.read_csv(pth(rel)).set_index("building_id")
        except FileNotFoundError:
            continue
        lbl.index = _normalize_index(lbl.index)

        common = feats.index.intersection(lbl.index)
        X = feats.loc[common]
        y = lbl.loc[common, "cluster"]

        m = _compute(X, y)
        rows.append(
            {
                "method": name,
                "n_clusters": int(y.nunique()),
                "n_buildings": int(len(common)),
                **m,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "method",
            "n_clusters",
            "n_buildings",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        ],
    )

    # --- updated here: support either "validation_metrics" or "metrics"
    out_key = "validation_metrics" if "validation_metrics" in cfg["outputs"] else "metrics"
    out_path = pth(cfg["outputs"][out_key])
    # ----------------------------------------

    out.to_csv(out_path, index=False)
    print(f"[metrics] wrote {out_path}")
    if not out.empty:
        print(out.to_string(index=False))

    # Reproducibility assertion for the paper's rounded GMM values
    try:
        t = pd.read_csv(out_path)
        gmm = t.loc[t["method"] == "gmm"]
        if gmm.empty:
            raise AssertionError("GMM row missing in validation_metrics.csv")
        sil = round(float(gmm["silhouette"].item()), 3)
        ch = round(float(gmm["calinski_harabasz"].item()), 1)
        db = round(float(gmm["davies_bouldin"].item()), 2)
        assert sil == 0.302, f"silhouette {sil} ≠ 0.302"
        assert ch == 12.2, f"calinski_harabasz {ch} ≠ 12.2"
        assert db == 1.07, f"davies_bouldin {db} ≠ 1.07"
        print("[metrics] paper check passed (GMM rounded values).")
    except Exception as e:
        sys.stderr.write(f"[metrics] paper check failed: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
