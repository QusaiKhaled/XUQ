#!/usr/bin/env python3
"""
Command-line pipeline: load data → clean → feature selection (MI) → train IT2-ANFIS → report metrics.

Run from the repository root, e.g.:
  python run_pipeline.py
  python run_pipeline.py --data data/raw/Data-Melbourne_F_fixed.csv --n-rules 7 --epochs 50
  python run_pipeline.py --data path/to/file.csv --no-train   # preprocessing + feature selection only
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_csv, clean_and_prepare
from src.features import select_features_mi
from src.config import (
    RAW_DIR,
    DEFAULT_CSV_NAME,
    DEFAULT_N_RULES,
    IT2_EPOCHS,
    IT2_PATIENCE,
    RANDOM_STATE,
)
from src.experiments.train import run_it2_training
from sklearn.metrics import r2_score, mean_squared_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the IT2-ANFIS pipeline: load CSV → clean → MI feature selection → train IT2-ANFIS → print test R² and MSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        type=Path,
        default=RAW_DIR / DEFAULT_CSV_NAME,
        help="Path to raw CSV (e.g. data/raw/Data-Melbourne_F_fixed.csv)",
    )
    p.add_argument(
        "--n-rules",
        type=int,
        default=DEFAULT_N_RULES,
        help="Number of IT2-ANFIS rules",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=IT2_EPOCHS,
        help="Max training epochs",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=IT2_PATIENCE,
        help="Early stopping patience",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random state for train/test split",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Only run preprocessing and feature selection; do not train the model",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose training output",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading data...")
    df = load_csv(data_path)
    print(f"  Raw shape: {df.shape}")

    print("Cleaning and preparing...")
    df_clean = clean_and_prepare(df)
    print(f"  Clean shape: {df_clean.shape}")

    print("Feature selection (mutual information)...")
    df_final = select_features_mi(df_clean)
    print(f"  Final shape: {df_final.shape}")

    if args.no_train:
        print("Done (--no-train: skipping training).")
        return

    X = df_final.drop(columns=["Energy Consumption"]).values
    y = df_final["Energy Consumption"].values

    print(f"Training IT2-ANFIS (n_rules={args.n_rules}, epochs={args.epochs}, patience={args.patience})...")
    model, (X_tr, X_val, X_test, y_tr, y_val, y_test) = run_it2_training(
        X,
        y,
        n_rules=args.n_rules,
        epochs=args.epochs,
        patience=args.patience,
        random_state=args.seed,
        verbose=not args.quiet,
    )

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("\n--- Test results ---")
    print(f"  R²:  {r2:.4f}")
    print(f"  MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
