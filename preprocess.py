import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

CM1_PATH = os.path.join(
    DATA_DIR,
    "cm1.csv"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features.pkl"
)

# Files for inspecting preprocessing
CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "cm1_cleaned.csv"
)

TRAIN_SCALED_PATH = os.path.join(
    DATA_DIR,
    "cm1_train_scaled.csv"
)

TEST_SCALED_PATH = os.path.join(
    DATA_DIR,
    "cm1_test_scaled.csv"
)


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():
    """
    Load the CM1 dataset.

    Paper 1 uses only cm1.csv.
    """

    if not os.path.exists(CM1_PATH):

        raise FileNotFoundError(
            f"CM1 dataset not found: {CM1_PATH}\n"
            "Make sure cm1.csv is inside the data folder."
        )

    data = pd.read_csv(
        CM1_PATH
    )

    print(
        f"CM1 shape: {data.shape}"
    )

    return data


# ============================================================
# TARGET PROCESSING
# ============================================================

def process_target(data):
    """
    Convert the defects column into binary labels.

    Output:
        0 = Non-defective
        1 = Defective
    """

    if "defects" not in data.columns:

        raise ValueError(
            "The dataset does not contain "
            "a 'defects' column."
        )

    target = data["defects"]

    # --------------------------------------------------------
    # Handle string labels
    # --------------------------------------------------------

    if target.dtype == object:

        target = (
            target
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0
        }

        target = target.map(
            mapping
        )

    else:

        target = pd.to_numeric(
            target,
            errors="coerce"
        )

    # --------------------------------------------------------
    # Check invalid values
    # --------------------------------------------------------

    if target.isna().any():

        invalid_count = target.isna().sum()

        raise ValueError(
            f"Found {invalid_count} invalid "
            "values in the 'defects' column."
        )

    target = target.astype(
        int
    )

    # --------------------------------------------------------
    # Check binary labels
    # --------------------------------------------------------

    unique_values = sorted(
        target.unique()
    )

    if not set(
        unique_values
    ).issubset({0, 1}):

        raise ValueError(
            f"Unexpected target values: "
            f"{unique_values}. "
            "Expected binary 0/1 labels."
        )

    return target


# ============================================================
# FEATURE IDENTIFICATION
# ============================================================

def get_feature_columns(data):
    """
    Identify the numerical software metrics.

    Excluded:
        id
        defects
    """

    excluded_columns = {
        "id",
        "defects"
    }

    feature_columns = []

    for column in data.columns:

        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):

            feature_columns.append(
                column
            )

    if len(feature_columns) == 0:

        raise ValueError(
            "No numerical feature columns "
            "were found."
        )

    return feature_columns


# ============================================================
# CLEAN FEATURES
# ============================================================

def clean_features(
    data,
    feature_columns
):
    """
    Clean the numerical software metrics.

    Steps:
        1. Convert to numeric
        2. Replace infinite values
        3. Replace missing values with median
    """

    X = data[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    for column in feature_columns:

        median_value = X[
            column
        ].median()

        if pd.isna(
            median_value
        ):

            median_value = 0.0

        X[column] = X[
            column
        ].fillna(
            median_value
        )

    return X


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    test_size=0.20,
    random_state=42
):
    """
    Complete preprocessing pipeline for CM1.

    Returns:

        X_train
        X_test
        y_train
        y_test
        feature_columns
        scaler
    """

    # --------------------------------------------------------
    # 1. Load CM1
    # --------------------------------------------------------

    data = load_dataset()

    # --------------------------------------------------------
    # 2. Process target
    # --------------------------------------------------------

    y = process_target(
        data
    )

    # --------------------------------------------------------
    # 3. Identify features
    # --------------------------------------------------------

    feature_columns = (
        get_feature_columns(
            data
        )
    )

    print(
        "\nFeatures used:"
    )

    for feature in feature_columns:

        print(
            "  -",
            feature
        )

    # --------------------------------------------------------
    # 4. Clean features
    # --------------------------------------------------------

    X = clean_features(
        data,
        feature_columns
    )

    # --------------------------------------------------------
    # Save cleaned dataset for inspection
    #
    # IMPORTANT:
    # This creates a NEW file.
    # cm1.csv remains untouched.
    # --------------------------------------------------------

    cleaned_data = X.copy()

    cleaned_data["defects"] = y.values

    cleaned_data.to_csv(
        CLEANED_DATA_PATH,
        index=False
    )

    print(
        "\nSaved cleaned dataset:",
        CLEANED_DATA_PATH
    )

    # --------------------------------------------------------
    # 5. Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    )

    print(
        "\nTrain samples:",
        len(X_train)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # --------------------------------------------------------
    # 6. Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Save scaled datasets for inspection
    #
    # These are NEW files.
    # --------------------------------------------------------

    train_scaled_df = pd.DataFrame(
        X_train,
        columns=feature_columns
    )

    train_scaled_df["defects"] = (
        y_train.values
    )

    train_scaled_df.to_csv(
        TRAIN_SCALED_PATH,
        index=False
    )

    test_scaled_df = pd.DataFrame(
        X_test,
        columns=feature_columns
    )

    test_scaled_df["defects"] = (
        y_test.values
    )

    test_scaled_df.to_csv(
        TEST_SCALED_PATH,
        index=False
    )

    print(
        "\nSaved scaled datasets:"
    )

    print(
        "  -",
        TRAIN_SCALED_PATH
    )

    print(
        "  -",
        TEST_SCALED_PATH
    )

    # --------------------------------------------------------
    # 7. Create model directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 8. Save scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # --------------------------------------------------------
    # 9. Save feature order
    # --------------------------------------------------------

    joblib.dump(
        feature_columns,
        FEATURES_PATH
    )

    print(
        "\nSaved:"
    )

    print(
        "Scaler:",
        SCALER_PATH
    )

    print(
        "Features:",
        FEATURES_PATH
    )

    # --------------------------------------------------------
    # 10. Convert labels to NumPy
    # --------------------------------------------------------

    y_train = y_train.to_numpy()

    y_test = y_test.to_numpy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        scaler
    )


# ============================================================
# PREPROCESS NEW INPUT
# ============================================================

def preprocess_new_input(
    input_data
):
    """
    Preprocess one new software-metric observation.

    input_data can be:

        dictionary

    or:

        pandas DataFrame
    """

    if not os.path.exists(
        SCALER_PATH
    ):

        raise FileNotFoundError(
            "Scaler not found. "
            "Please train the model first."
        )

    if not os.path.exists(
        FEATURES_PATH
    ):

        raise FileNotFoundError(
            "Feature list not found. "
            "Please train the model first."
        )

    # --------------------------------------------------------
    # Load preprocessing objects
    # --------------------------------------------------------

    scaler = joblib.load(
        SCALER_PATH
    )

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    # --------------------------------------------------------
    # Convert dictionary to DataFrame
    # --------------------------------------------------------

    if isinstance(
        input_data,
        dict
    ):

        input_data = pd.DataFrame(
            [input_data]
        )

    elif not isinstance(
        input_data,
        pd.DataFrame
    ):

        raise TypeError(
            "input_data must be a dictionary "
            "or pandas DataFrame."
        )

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in input_data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            + ", ".join(
                missing_features
            )
        )

    # --------------------------------------------------------
    # Keep EXACT feature order
    # --------------------------------------------------------

    X = input_data[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        0
    )

    # --------------------------------------------------------
    # Apply training scaler
    # --------------------------------------------------------

    X = scaler.transform(
        X
    )

    return X


# ============================================================
# DATASET INFORMATION
# ============================================================

def print_dataset_information():

    data = load_dataset()

    print(
        "\n" + "=" * 60
    )

    print(
        "CM1 DATASET INFORMATION"
    )

    print(
        "=" * 60
    )

    print(
        "\nTotal rows:",
        len(data)
    )

    print(
        "\nColumns:"
    )

    for column in data.columns:

        print(
            "  -",
            column
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    if "defects" in data.columns:

        target = process_target(
            data
        )

        print(
            "\nDefect distribution:"
        )

        print(
            "Non-defective (0):",
            int(
                (target == 0).sum()
            )
        )

        print(
            "Defective (1):",
            int(
                (target == 1).sum()
            )
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_dataset_information()

    print(
        "\nPreparing CM1 training data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        scaler
    ) = prepare_data()

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nX_train shape:",
        X_train.shape
    )

    print(
        "X_test shape:",
        X_test.shape
    )

    print(
        "y_train shape:",
        y_train.shape
    )

    print(
        "y_test shape:",
        y_test.shape
    )

    print(
        "\nNumber of features:",
        len(feature_columns)
    )

    print(
        "\nFeature order:"
    )

    print(
        feature_columns
    )