# Reproducible code and dataset for EPC-based clustering

This repository contains the anonymized dataset and code used in the paper  
“Feature extraction-based clustering selection methodology to identify representative buildings for scalable energy simulations.”  
The goal is to allow anyone to reproduce the main results and extend the analysis.

## Structure

- code/steps/ contains the main pipeline scripts  
- config/config.yaml holds parameters and paths  
- data/processed/ contains the input CSV files  
- outputs/ is created automatically when the pipeline runs  

## Data inputs

The following files are included in data/processed:

- epc_metadata.csv: EPC-based building attributes (energy performance value, weighted U-value, air leakage, heat recovery)  
- features_epc_only_standardized.csv: standardized feature table used for clustering  
- data_dictionary.xlsx: description of all variables  

The data are anonymized, with no school names or addresses.

## How to run

1. Build the features parquet  
   ```bash
   python3 code/steps/00_features_from_vs_csv.py
   ```

2. Run EPC clustering (K-Medoids, Agglomerative, GMM)  
   ```bash
   python3 code/steps/01_cluster_with_custom_impl.py
   ```

3. Compute internal validation metrics  
   ```bash
   python3 code/steps/02_validation_metrics.py
   ```

4. Compute pairwise agreement between methods  
   ```bash
   python3 code/steps/03_pairwise_agreement.py
   ```

Outputs are written into the outputs/ folder as set in config/config.yaml.

## Expected outputs

- validation_metrics.csv: clustering quality indices (Silhouette, DB, CH)  
- agreement_metrics.csv: ARI and NMI comparisons  
- clusters_kmedoids.csv: EPC-based labels from K-Medoids  
- clusters_agglomerative.csv: EPC-based labels from Agglomerative clustering  
- clusters_gmm.csv: EPC-based labels from Gaussian Mixture Model  
- features.parquet: standardized EPC features used for clustering  

These reproduce the tables reported in the paper.

## Requirements

Install the dependencies with:  
```bash
pip install -r requirements.txt
```

Main packages are scikit-learn, numpy, pandas, scipy, and pyyaml.

## License

Code is released under the MIT License.  
Data is released under the Creative Commons Attribution 4.0 license (CC-BY-4.0).
