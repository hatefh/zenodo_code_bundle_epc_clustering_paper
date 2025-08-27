import pandas as pd
import yaml
from code.utils.paths import pth
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def load_cfg():
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)

def compute_metrics(X, labels):
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }

def main():
    cfg = load_cfg()

    # Load EPC feature matrix used for clustering (as per paper)
    feats = pd.read_parquet(pth(cfg["outputs"]["features"])).set_index("building_id")
    # No need to drop anything if features file already contains only EPC features

    methods = [
        ("kmedoids",       cfg["outputs"]["clusters_kmedoids"]),
        ("agglomerative",  cfg["outputs"]["clusters_agglomerative"]),
        ("gmm",            cfg["outputs"]["clusters_gmm"]),
    ]

    rows = []
    for method_name, relpath in methods:
        try:
            lbl_df = pd.read_csv(pth(relpath)).set_index("building_id")
        except FileNotFoundError:
            # skip methods not run
            continue

        # Align features and labels
        common_ids = feats.index.intersection(lbl_df.index)
        X = feats.loc[common_ids].values
        y = lbl_df.loc[common_ids, "cluster"].values

        # Compute metrics
        m = compute_metrics(X, y)
        m.update({
            "method": method_name,
            "n_clusters": int(pd.Series(y).nunique()),
            "n_buildings": int(len(common_ids)),
        })
        rows.append(m)

    out = pd.DataFrame(rows, columns=[
        "method", "n_clusters", "n_buildings",
        "silhouette", "davies_bouldin", "calinski_harabasz"
    ])

    out_path = pth(cfg["outputs"]["metrics"])
    out.to_csv(out_path, index=False)
    print(f"Validation metrics saved to: {out_path}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()

