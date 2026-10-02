import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from code2vec_extractor import extractor


# ============================================================
# PATHS & CONSTANTS
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

CM1_PATH = os.path.join(DATA_DIR, "cm1.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cm1_cleaned.csv")
TRAIN_SCALED_PATH = os.path.join(DATA_DIR, "cm1_train_scaled.csv")
TEST_SCALED_PATH = os.path.join(DATA_DIR, "cm1_test_scaled.csv")


# ============================================================
# DATASET LOADING & CLEANING (SECTION 4.2.1)
# ============================================================

def load_dataset(dataset_path=None):
    """
    Loads software defect dataset (e.g. CM1 from PROMISE repository).
    Section 4.1 Data Collection.
    """
    target_path = dataset_path or CM1_PATH
    if not os.path.exists(target_path):
        # Fallback to root if data directory not populated
        if os.path.exists("cm1.csv"):
            target_path = "cm1.csv"
        else:
            raise FileNotFoundError(
                f"Dataset not found at: {target_path}. Please place cm1.csv in data/ or root directory."
            )

    data = pd.read_csv(target_path)
    print(f"Loaded dataset from {target_path} with shape: {data.shape}")
    return data


def process_target(data):
    """
    Converts target defect labels into binary format (0 = Non-defective, 1 = Defective).
    Handles boolean, string, and integer representations.
    """
    target_col = None
    for col in ["defects", "Defective", "bug", "fault", "class"]:
        if col in data.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError("Could not find a defect target column (e.g. 'defects') in dataset.")

    target = data[target_col]

    if target.dtype == object or target.dtype == bool:
        target = target.astype(str).str.strip().str.lower()
        mapping = {
            "true": 1, "false": 0, "yes": 1, "no": 0,
            "1": 1, "0": 0, "buggy": 1, "clean": 0,
            "defective": 1, "non-defective": 0
        }
        target = target.map(mapping)
    else:
        target = pd.to_numeric(target, errors="coerce")

    # Binarize if count of defects > 0
    target = target.apply(lambda v: 1 if v > 0 else 0)

    if target.isna().any():
        target = target.fillna(0)

    return target.astype(int)


def get_feature_columns(data):
    """
    Identifies software metric columns excluding ID and label columns.
    Section 4.1: 20+ traditional static metrics (McCabe & Halstead).
    """
    excluded = {"id", "defects", "Defective", "bug", "fault", "class", "name", "file", "version"}
    feature_cols = [c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]
    if not feature_cols:
        # Fallback: try converting any remaining non-excluded columns to numeric
        feature_cols = [c for c in data.columns if c not in excluded]
    return feature_cols


def clean_features(data, feature_columns, missing_threshold=0.5):
    """
    Handling Missing Values (Section 4.2.1):
    - Replaces inf/-inf with NaN
    - Trims features with disproportionately high missing values (> missing_threshold)
    - Imputes missing numerical values using feature median
    """
    X = data[feature_columns].copy()

    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Filter out columns with excessive missing values
    valid_cols = []
    for col in feature_columns:
        missing_rate = X[col].isna().mean()
        if missing_rate < missing_threshold:
            valid_cols.append(col)
        else:
            print(f"Trimming feature '{col}' due to high missing rate: {missing_rate:.2%}")

    X = X[valid_cols]

    # Median imputation
    for col in valid_cols:
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        X[col] = X[col].fillna(med)

    return X, valid_cols


# ============================================================
# DATA PREPARATION PIPELINE
# ============================================================

def prepare_data(test_size=0.20, random_state=42):
    """
    Complete preprocessing pipeline:
    1. Load data
    2. Extract & binarize labels
    3. Clean and impute missing metric values
    4. Stratified Train-Test split
    5. StandardScaler Normalization (Section 4.2.2)
    6. Persist artifacts
    """
    data = load_dataset()
    y = process_target(data)
    feature_cols = get_feature_columns(data)
    X, cleaned_feature_cols = clean_features(data, feature_cols)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save cleaned inspection copy
    cleaned_df = X.copy()
    cleaned_df["defects"] = y.values
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Feature Normalization and Scaling (Section 4.2.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaled datasets
    pd.DataFrame(X_train_scaled, columns=cleaned_feature_cols).assign(defects=y_train.values).to_csv(TRAIN_SCALED_PATH, index=False)
    pd.DataFrame(X_test_scaled, columns=cleaned_feature_cols).assign(defects=y_test.values).to_csv(TEST_SCALED_PATH, index=False)

    # Save scaler and feature list
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(cleaned_feature_cols, FEATURES_PATH)

    print(f"\nPrepared dataset: {len(X_train)} training samples, {len(X_test)} testing samples, {len(cleaned_feature_cols)} features.")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_test.to_numpy(),
        cleaned_feature_cols,
        scaler
    )


def preprocess_new_input(input_data):
    """
    Preprocesses new software metric observation for prediction.
    """
    if not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Preprocessing artifacts (scaler.pkl / features.pkl) not found. Run train.py first.")

    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a dict or pandas DataFrame.")

    # Fill missing features with 0.0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaled_X = scaler.transform(X)
    return scaled_X


def preprocess_code_snippet(code_str):
    """
    Converts raw source code snippet into dense Code2Vec vector representation (Section 4.3).
    Returns vector (128-dim) and extracted AST path contexts.
    """
    vec, contexts = extractor.extract_code_vector(code_str)
    return vec, contexts


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, scaler = prepare_data()
    print("Preprocessing completed successfully!")