import pandas as pd
import yaml
from code.utils.paths import pth
from sklearn.mixture import GaussianMixture
import numpy as np
np.random.seed(42)

def load_cfg():
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)

def main(k=4):
    cfg = load_cfg()
    X = pd.read_parquet(pth(cfg["outputs"]["features"])).set_index("building_id")
    model = GaussianMixture(n_components=k, random_state=42, n_init=10)
    labels = model.fit_predict(X.values)
    out = pd.DataFrame({"building_id": X.index, "cluster": labels})
    out.to_csv(pth(cfg["outputs"]["clusters_gmm"]), index=False)
    print(f"GMM clustering finished with k={k}, results saved.")

if __name__ == "__main__":
    main()
