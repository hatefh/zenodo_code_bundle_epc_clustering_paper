"""Step 00: Materialize the VS Code standardized features CSV into a parquet file.

Reads the CSV defined in config.yaml (data_dir + inputs.features_csv),
ensures a 'building_id' index, and writes outputs/features.parquet
for downstream steps to consume.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import yaml
from code.utils.paths import pth


def load_cfg() -> dict:
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_cfg()

    data_dir = Path(cfg.get("data_dir", "."))
    rel_csv = cfg["inputs"]["features_csv"]
    src = (data_dir / rel_csv).resolve()
    dst = Path(pth(cfg["outputs"]["features"]))

    if not src.exists():
        sys.stderr.write(
            f"[features] input CSV not found: {src}\n"
            "Check config.yaml: data_dir + inputs.features_csv\n"
        )
        return 1

    df = pd.read_csv(src, index_col=0)

    # Try to coerce building identifier
    if df.index.name is None or str(df.index.name).lower() not in {"building_id", "id", "no.", "no"}:
        for cand in ("building_id", "id", "no.", "no"):
            if cand in df.columns:
                df = df.set_index(cand)
                break

    df.index = df.index.astype(str).str.strip()
    df.index.name = "building_id"

    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].unique().tolist()
        sys.stderr.write(f"[features] duplicate building_id values: {dupes[:5]}...\n")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(dst, index=False)
    print(f"[features] wrote {dst}  shape={df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
