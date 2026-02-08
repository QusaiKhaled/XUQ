"""Default configuration and constants for the IT2-ANFIS pipeline."""

from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data
DEFAULT_CSV_NAME = "Data-Melbourne_F_fixed.csv"
TARGET_COLUMN = "Energy Consumption"
DROP_COLUMNS = ["Unnamed: 0"]

# Feature selection
MI_RANDOM_STATE = 0
MIN_MI_THRESHOLD = 0.0  # Keep features with MI > this (0 = drop only exactly zero)

# Train/val/test
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # of training set
RANDOM_STATE = 42

# IT2-ANFIS defaults
DEFAULT_N_RULES = 7
IT2_M1 = 1.5
IT2_M2 = 3.0
IT2_MIN_SIGMA = 0.05
IT2_EPOCHS = 100
IT2_PATIENCE = 30
IT2_BATCH_SIZE = 64
IT2_FREEZE_ANTECEDENTS = True
IT2_LAMBDA_L1 = 0.005
IT2_LAMBDA_L2 = 0.01
IT2_Q = 0.5

# Rule sweep
RULE_SWEEP_MIN = 1
RULE_SWEEP_MAX = 50
RULE_SWEEP_N_SEEDS = 10
