# code/steps/01_cluster_with_custom_impl.py
"""Step 01: Run clustering (custom K-Medoids, Agglomerative, GMM) on EPC features.

Reads outputs/features.parquet and writes three label CSVs defined in config.yaml.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from code.utils.paths import pth


def load_cfg() -> dict:
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


# --- K-Medoids (same behavior as your original code) ---
def k_medoids(X: np.ndarray, k: int, max_iter: int = 300, random_state: int = 42):
    np.random.seed(random_state)
    m = X.shape[0]
    medoid_indices = np.random.choice(m, k, replace=False)
    medoids = X[medoid_indices]

    for _ in range(max_iter):
        distances = cdist(X, medoids, metric="euclidean")
        labels = np.argmin(distances, axis=1)

        new_medoids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue
            dist_sum = cdist(cluster_points, cluster_points).sum(axis=1)
            new_medoids.append(cluster_points[np.argmin(dist_sum)])
        new_medoids = np.array(new_medoids)

        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids

    final_indices = []
    for medoid in medoids:
        distances = np.linalg.norm(X - medoid, axis=1)
        final_indices.append(np.argmin(distances))
    return labels, final_indices


def _fit_agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def _fit_gmm(X: np.ndarray, k: int, seed: int, cov_type: str, init_params: str,
             n_init: int, max_iter: int, tol: float, reg_covar: float) -> np.ndarray:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        random_state=seed,
        init_params=init_params,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        reg_covar=reg_covar,
    )
    return gmm.fit_predict(X)


def _write_labels(building_id: np.ndarray, labels: np.ndarray, out_path: Path):
    df = pd.DataFrame({"building_id": building_id.astype(str), "cluster": labels})
    df.to_csv(out_path, index=False)


def main() -> int:
    cfg = load_cfg()

    # config
    k = int(cfg.get("clustering", {}).get("k", 4))
    seeds = cfg.get("clustering", {}).get("seeds", {})
    rs_global = int(cfg.get("random_seed", 42))
    rs_kmedoids = int(seeds.get("kmedoids", rs_global))
    rs_gmm = int(seeds.get("gmm", 0))

    gmm_cfg = cfg.get("clustering", {}).get("gmm", {})
    cov_type  = gmm_cfg.get("covariance_type", "full")
    init_par  = gmm_cfg.get("init_params", "kmeans")
    n_init    = int(gmm_cfg.get("n_init", 1))
    max_iter  = int(gmm_cfg.get("max_iter", 100))
    tol       = float(gmm_cfg.get("tol", 1e-3))
    reg_covar = float(gmm_cfg.get("reg_covar", 1e-6))

    # ensure outputs dir exists
    Path(pth("outputs")).mkdir(parents=True, exist_ok=True)

    # --- EPC features ---
    feats = pd.read_parquet(pth(cfg["outputs"]["features"]))
    if "building_id" not in feats.columns:
        raise KeyError("features parquet is missing 'building_id' column")
    ids = feats["building_id"].astype(str).str.strip().values
    X = feats.drop(columns=["building_id"]).values

    # K-Medoids
    km_labels, _ = k_medoids(X, k=k, random_state=rs_kmedoids)
    _write_labels(ids, km_labels, pth(cfg["outputs"]["clusters_kmedoids"]))

    # Agglomerative
    ag_labels = _fit_agglomerative(X, k)
    _write_labels(ids, ag_labels, pth(cfg["outputs"]["clusters_agglomerative"]))

    # GMM
    gm_labels = _fit_gmm(X, k, rs_gmm, cov_type, init_par, n_init, max_iter, tol, reg_covar)
    _write_labels(ids, gm_labels, pth(cfg["outputs"]["clusters_gmm"]))

    print("[cluster] wrote EPC cluster labels to outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
