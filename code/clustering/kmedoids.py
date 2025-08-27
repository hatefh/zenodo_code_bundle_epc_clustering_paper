import pandas as pd
import yaml
from code.utils.paths import pth
from sklearn_extra.cluster import KMedoids
import numpy as np
np.random.seed(42)  # ensures reproducibility

def load_cfg():
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)

def main(k=4):
    cfg = load_cfg()
    X = pd.read_parquet(pth(cfg["outputs"]["features"])).set_index("building_id")
    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X.values)
    out = pd.DataFrame({"building_id": X.index, "cluster": labels})
    out.to_csv(pth(cfg["outputs"]["clusters_kmedoids"]), index=False)
    print(f"K-Medoids clustering finished with k={k}, results saved.")

if __name__ == "__main__":
    main()
