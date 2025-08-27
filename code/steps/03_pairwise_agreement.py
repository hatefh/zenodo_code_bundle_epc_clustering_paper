"""Step 03: Pairwise agreement between methods (ARI/NMI)."""
from __future__ import annotations

import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from code.utils.paths import pth


PAPER = {
    ("kmedoids", "agglomerative"): {"ARI": 0.656, "NMI": 0.768},
    ("kmedoids", "gmm"): {"ARI": 0.215, "NMI": 0.507},
    ("agglomerative", "gmm"): {"ARI": 0.303, "NMI": 0.520},
}
TOL = {"ARI": 0.005, "NMI": 0.005}


def load_cfg() -> dict:
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def _norm_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["building_id"] = df["building_id"].astype(str).str.strip()
    return df.set_index("building_id")


def main() -> int:
    cfg = load_cfg()
    k = _norm_index(pd.read_csv(pth(cfg["outputs"]["clusters_kmedoids"])))
    a = _norm_index(pd.read_csv(pth(cfg["outputs"]["clusters_agglomerative"])))
    g = _norm_index(pd.read_csv(pth(cfg["outputs"]["clusters_gmm"])))

    common = k.index.intersection(a.index).intersection(g.index)
    k, a, g = k.loc[common], a.loc[common], g.loc[common]

    pairs = [
        ("K-Medoids vs Agglomerative", "kmedoids", "agglomerative", k["cluster"].values, a["cluster"].values),
        ("K-Medoids vs GMM", "kmedoids", "gmm", k["cluster"].values, g["cluster"].values),
        ("Agglomerative vs GMM", "agglomerative", "gmm", a["cluster"].values, g["cluster"].values),
    ]

    rows = []
    for title, m1, m2, y1, y2 in pairs:
        ari = adjusted_rand_score(y1, y2)
        nmi = normalized_mutual_info_score(y1, y2)
        rows.append({"MethodPair": title, "ARI": ari, "NMI": nmi})

        tgt = PAPER.get((m1, m2))
        if tgt:
            ok_ari = abs(ari - tgt["ARI"]) <= TOL["ARI"]
            ok_nmi = abs(nmi - tgt["NMI"]) <= TOL["NMI"]
            print(f"{title}: ARI={ari:.3f} (target {tgt['ARI']:.3f}) "
                  f"{'OK' if ok_ari else 'DIFF'}, "
                  f"NMI={nmi:.3f} (target {tgt['NMI']:.3f}) "
                  f"{'OK' if ok_nmi else 'DIFF'}")

    out = pd.DataFrame(rows, columns=["MethodPair", "ARI", "NMI"])
    out_path = pth("outputs", "agreement_metrics.csv")
    out.to_csv(out_path, index=False)
    print(f"[agreement] wrote {out_path}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
