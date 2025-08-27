#!/usr/bin/env python3
"""
Compute agreement between DH-usage-based clustering and EPC-based clustering
for K-Medoids, Agglomerative, and GMM. Outputs a CSV table with ARI and NMI.

Expected inputs (CSV), one row per building:
  - EPC labels (produced from EPC features)
  - DH labels (produced from DH usage)

Each CSV should have either:
  a) columns ['building_id', 'cluster']  OR
  b) an index that is building_id and a column named 'cluster' (or similar)

Config:
- Reads config/config.yaml.
- Uses outputs.* paths for EPC labels (clusters_*).
- Uses outputs.dh_clusters_* paths for DH labels.

Result:
- outputs/agreement_dh_vs_epc.csv with columns:
  Method, ARI_vs_EPC, NMI_vs_EPC
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


HERE = Path(__file__).resolve().parent.parent.parent  # repo root = bundle root
CFG = HERE / "config" / "config.yaml"


def pth(rel: str | os.PathLike) -> Path:
    """Resolve a path relative to repo root."""
    return (HERE / rel).resolve()


def _read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_labels(csv_path: Path) -> pd.Series:
    """Load a CSV of labels and return a Series indexed by building_id.
    Accepts several common shapes:
      - columns: ['building_id','cluster']
      - any two columns where one looks like an id and one like labels
      - index as id and a single label column
    """
    df = pd.read_csv(csv_path)
    # Try common column names
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    # Guess building id column
    id_col = None
    for c in ("building_id", "id", "building"):
        if c in df.columns:
            id_col = c
            break
    if id_col is None and df.index.name and df.index.name.lower() in ("building_id", "id", "building"):
        id_col = df.index.name.lower()

    # Guess cluster column
    lab_col = None
    for c in ("cluster", "label", "labels"):
        if c in df.columns:
            lab_col = c
            break
    if lab_col is None:
        # if only two columns, use the non-id one
        if len(df.columns) == 2 and id_col in df.columns:
            lab_col = [c for c in df.columns if c != id_col][0]

    # Normalize to Series
    if id_col in df.columns:
        s = df.set_index(id_col)[lab_col]
    elif id_col == df.index.name and lab_col in df.columns:
        s = df[lab_col]
    else:
        raise ValueError(f"Could not infer id/label columns in {csv_path}")

    # Make sure deterministic type
    return s.astype("int64").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG), help="Path to config.yaml")
    args = ap.parse_args()

    cfg = _read_yaml(Path(args.config))

    # Where to save results
    out_csv = pth(cfg["outputs"].get("agreement_metrics", "outputs/agreement_dh_vs_epc.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # EPC-based labels (produced by Step 01)
    epc_paths = {
        "K-Medoids": pth(cfg["outputs"]["clusters_kmedoids"]),
        "Agglomerative": pth(cfg["outputs"]["clusters_agglomerative"]),
        "GMM": pth(cfg["outputs"]["clusters_gmm"]),
    }

    # DH-usage-based labels (you should generate these separately from DH data).
    # Add these keys to your config.yaml under outputs:
    #   dh_clusters_kmedoids, dh_clusters_agglomerative, dh_clusters_gmm
    dh_cfg = cfg["outputs"]
    try:
        dh_paths = {
            "K-Medoids": pth(dh_cfg["dh_clusters_kmedoids"]),
            "Agglomerative": pth(dh_cfg["dh_clusters_agglomerative"]),
            "GMM": pth(dh_cfg["dh_clusters_gmm"]),
        }
    except KeyError as e:
        raise SystemExit(
            f"Missing '{e.args[0]}' in config.outputs. "
            "Add dh_clusters_* paths for the DH-derived labels."
        )

    rows = []
    for method in ("K-Medoids", "Agglomerative", "GMM"):
        epc_csv = epc_paths[method]
        dh_csv = dh_paths[method]
        if not epc_csv.exists():
            raise FileNotFoundError(f"EPC labels not found: {epc_csv}")
        if not dh_csv.exists():
            raise FileNotFoundError(f"DH labels not found: {dh_csv}")

        epc = _load_labels(epc_csv)
        dh = _load_labels(dh_csv)

        # Align by building_id
        common = epc.index.intersection(dh.index)
        epc = epc.loc[common]
        dh = dh.loc[common]

        ari = adjusted_rand_score(dh.values, epc.values)
        nmi = normalized_mutual_info_score(dh.values, epc.values, average_method="arithmetic")

        rows.append({"Method": method, "ARI vs EPC": ari, "NMI vs EPC": nmi})

    out = pd.DataFrame(rows)
    # Optional rounding to match table presentation
    out_rounded = out.copy()
    out_rounded["ARI vs EPC"] = out_rounded["ARI vs EPC"].round(3)
    out_rounded["NMI vs EPC"] = out_rounded["NMI vs EPC"].round(3)

    out.to_csv(out_csv, index=False)
    print("\nAgreement between DH-usage and EPC-based clusters:")
    print(out_rounded.to_string(index=False))
    print(f"\nSaved full-precision results to: {out_csv}")


if __name__ == "__main__":
    main()
