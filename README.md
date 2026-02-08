# Interval Type-2 ANFIS for Uncertainty Quantification

This repository implements an **Interval Type-2 Adaptive Neuro-Fuzzy Inference System (IT2-ANFIS)** for regression and uncertainty quantification, applied to **wastewater treatment plant energy consumption** prediction. It accompanies the paper [*Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2 Neuro-Fuzzy System*](https://doi.org/10.48550/arXiv.2601.18897) (Khaled et al., 2026).

## Objective

- Predict daily energy consumption of a full-scale wastewater treatment plant from hydraulic, biological, and climate features.
- Provide **uncertainty bounds** (lower/upper prediction intervals) via the IT2 fuzzy framework.
- Compare against baselines: Random Forest, SVM, zero-order ANFIS, first-order ANFIS.
- Study the effect of **number of rules** (1–50) on MSE for both first-order ANFIS and IT2-ANFIS with multiple random seeds.

## Dataset

**Full-Scale Wastewater Treatment Plant Data**  
Energy consumption, climate, and wastewater characteristics of Melbourne eastern wastewater treatment plant (2014–2019).

- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/pprkvz3vbd/1)  
- **Dataset citation**: Bagehrzadeh, Faramarz (2021), “Full Scale Wastewater Treatment Plant Data”, Mendeley Data, V1, [doi: 10.17632/pprkvz3vbd.1](https://doi.org/10.17632/pprkvz3vbd.1)  
- **Paper (use of data)**: [DOI: 10.1016/j.psep.2021.08.040](https://doi.org/10.1016/j.psep.2021.08.040)

## Repository Structure

```
it2-anfis/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── run_pipeline.py      # CLI: load → clean → feature selection → train IT2-ANFIS
├── data/
│   ├── raw/           # Original CSVs (use git-lfs if large)
│   ├── interim/       # Cleaned, merged datasets
│   └── processed/     # Final feature matrices, train/test splits
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_it2_anfis_training.ipynb
│   ├── 04_baselines_and_evaluation.ipynb
│   ├── 05_rule_sweep_experiments.ipynb
│   └── 99_results_and_figures.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── selection.py
│   ├── models/
│   │   ├── it2_anfis.py
│   │   ├── anfis_zero.py
│   │   ├── anfis_first.py
│   │   ├── rf_model.py
│   │   └── svm_model.py
│   ├── experiments/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── sweep.py
│   ├── viz/
│   │   └── plot.py
│   └── utils/
│       ├── metrics.py
│       └── persistence.py
├── models/            # Trained model artifacts (gitignored)
├── figures/           # Generated figures
├── results/           # CSV/JSON experiment summaries
├── docs/
├── tests/
│   ├── test_data.py
│   └── test_models.py
└── .github/workflows/ # Optional CI
```

## Quick Start

### Environment

```bash
# Option 1: pip
pip install -r requirements.txt

# Option 2: conda
conda env create -f environment.yml
conda activate it2-anfis
```

### Data

Place the raw CSV (e.g. `Data-Melbourne_F_fixed.csv`) in `data/raw/` or set the path in `src/config.py`.

### Command-line pipeline (`run_pipeline.py`)

From the repository root you can run the full pipeline (load → clean → feature selection → train IT2-ANFIS) in one go:

```bash
# Default: use data/raw/Data-Melbourne_F_fixed.csv, 7 rules, 100 epochs
python run_pipeline.py

# Custom data path and model settings
python run_pipeline.py --data data/raw/Data-Melbourne_F_fixed.csv --n-rules 7 --epochs 50

# Preprocessing and feature selection only (no training)
python run_pipeline.py --no-train

# Less verbose training output
python run_pipeline.py --quiet
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Path to raw CSV | `data/raw/Data-Melbourne_F_fixed.csv` |
| `--n-rules` | Number of IT2-ANFIS rules | 7 |
| `--epochs` | Max training epochs | 100 |
| `--patience` | Early stopping patience | 30 |
| `--seed` | Random state for train/test split | 42 |
| `--no-train` | Only run load + clean + feature selection | off |
| `--quiet` | Reduce training log output | off |

The script prints the cleaned and final dataframe shapes, then after training the test **R²** and **MSE**.

### Run pipeline (notebooks)

1. **Preprocess**: `notebooks/01_data_preprocessing.ipynb` — load CSV, clean, remove NaNs, show clean `df` head and shape.
2. **Feature selection**: `notebooks/02_feature_selection.ipynb` — mutual information, visualize selected features, final `df` head, target density.
3. **IT2-ANFIS**: `notebooks/03_it2_anfis_training.ipynb` — train IT2-ANFIS (e.g. 7 rules), true vs predicted with R², membership function plots (upper/lower bounds).
4. **Baselines**: `notebooks/04_baselines_and_evaluation.ipynb` — Random Forest, zero-order ANFIS, first-order ANFIS, SVM.
5. **Rule sweep**: `notebooks/05_rule_sweep_experiments.ipynb` — MSE vs rules (1–50) for first-order ANFIS and IT2-ANFIS, 10 seeds per rule count, side-by-side plots.
6. **Results**: `notebooks/99_results_and_figures.ipynb` — aggregate results and figures.

### Programmatic Training

```python
from src.data.load_data import load_csv
from src.data.preprocessing import clean_and_prepare
from src.features.selection import select_features_mi
from src.models.it2_anfis import IT2_TSK_ANFIS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = load_csv("data/raw/Data-Melbourne_F_fixed.csv")
df_clean = clean_and_prepare(df)
df_final = select_features_mi(df_clean, target="Energy Consumption", min_mi=0)

X = df_final.drop(columns=["Energy Consumption"]).values
y = df_final["Energy Consumption"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = IT2_TSK_ANFIS(n_rules=7, X=X_tr)
model.fit(X_tr, y_tr, X_val, y_val, epochs=100, patience=30)
y_pred = model.predict(X_test)
print("Test R²:", r2_score(y_test, y_pred))
```

## Model Summary

- **IT2-ANFIS**: Interval Type-2 TSK ANFIS with Gaussian membership functions (lower/upper), grid or random-grid initialization, mini-batch training, optional antecedent tuning via PyTorch autodiff, fixed design factor `q=0.5`.
- **Explainability**: `model.explain_prediction(X_input)` prints rule contributions and interval widths.
- **Visualization**: Trained membership functions (upper/lower bands) in `src/viz/plot.py`.



## Citation

This repository accompanies the paper:

**Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2 Neuro-Fuzzy System**  
Qusai Khaled, Bahjat Mallak, Uzay Kaymak, Laura Genga  
*arXiv preprint arXiv:2601.18897*, 2026.  
**DOI:** [10.48550/arXiv.2601.18897](https://doi.org/10.48550/arXiv.2601.18897)

If you use this code or the methodology, please cite:

```bibtex
@article{khaled2026explainable,
  title={Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2 Neuro-Fuzzy System},
  author={Khaled, Qusai and Mallak, Bahjat and Kaymak, Uzay and Genga, Laura},
  journal={arXiv preprint arXiv:2601.18897},
  year={2026}
}
```

For the dataset, please also cite the data source and paper linked in the [Dataset](#dataset) section.
