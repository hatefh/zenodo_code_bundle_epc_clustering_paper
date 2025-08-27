#!/usr/bin/env python3

import sys, json, platform

def section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)

def ok(msg):
    print(f"[OK] {msg}")

def fail(msg, e=None):
    print(f"[FAIL] {msg}")
    if e:
        print(f"       {type(e).__name__}: {e}")
    sys.exit(1)

def main():
    section("Python & OS")
    print("Python:", sys.version.split()[0])
    print("OS:", platform.platform())

    section("Package versions")
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import sklearn_extra
        import yaml
        import pyarrow
        print("numpy:", np.__version__)
        print("pandas:", pd.__version__)
        print("scikit-learn:", sklearn.__version__)
        print("scikit-learn-extra:", sklearn_extra.__version__)
        print("pyyaml:", yaml.__version__)
        print("pyarrow:", pyarrow.__version__)
        ok("Imports OK")
    except Exception as e:
        fail("Package import failed", e)

    section("Quick clustering smoke tests")
    import numpy as np
    X = np.random.RandomState(42).rand(20, 4)

    # KMedoids
    try:
        from sklearn_extra.cluster import KMedoids
        _ = KMedoids(n_clusters=3, random_state=42).fit_predict(X)
        ok("KMedoids")
    except Exception as e:
        fail("KMedoids failed", e)

    # Agglomerative
    try:
        from sklearn.cluster import AgglomerativeClustering
        _ = AgglomerativeClustering(n_clusters=3).fit_predict(X)
        ok("AgglomerativeClustering")
    except Exception as e:
        fail("AgglomerativeClustering failed", e)

    # GMM
    try:
        from sklearn.mixture import GaussianMixture
        _ = GaussianMixture(n_components=3, random_state=42).fit_predict(X)
        ok("GaussianMixture")
    except Exception as e:
        fail("GaussianMixture failed", e)

    section("Summary")
    ok("All checks passed. Your environment is ready.")

if __name__ == "__main__":
    main()
