import os
import joblib
import torch

from model import HybridDefectModel
from preprocess import preprocess_new_input


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "defect_model.pth"
)

FEATURES_PATH = os.path.join(
    "models",
    "features.pkl"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Please run train.py first."
        )

    device = get_device()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    num_features = checkpoint[
        "num_features"
    ]

    linknet_weight = checkpoint.get(
        "linknet_weight",
        0.5
    )

    bilstm_weight = checkpoint.get(
        "bilstm_weight",
        0.5
    )

    model = HybridDefectModel(

        num_features=num_features,

        linknet_weight=linknet_weight,

        bilstm_weight=bilstm_weight
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    return (
        model,
        device
    )


# ============================================================
# PREDICT ONE SAMPLE
# ============================================================

def predict(input_data):

    """
    Predict whether a software instance is defective.

    Parameters
    ----------
    input_data : dict or pandas.DataFrame

        Software metric values.

    Returns
    -------
    dict

        Prediction and probability information.
    """

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model, device = load_model()

    # --------------------------------------------------------
    # Preprocess input
    # --------------------------------------------------------

    processed_input = (
        preprocess_new_input(
            input_data
        )
    )

    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------------

    X = torch.tensor(
        processed_input,
        dtype=torch.float32
    )

    X = X.to(
        device
    )

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.predict(
            X
        )

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    prediction = int(
        outputs[
            "prediction"
        ]
        .cpu()
        .numpy()[0]
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        outputs[
            "confidence"
        ]
        .cpu()
        .numpy()[0]
    )

    # --------------------------------------------------------
    # Final fused probabilities
    # --------------------------------------------------------

    probabilities = (
        outputs[
            "probabilities"
        ]
        .cpu()
        .numpy()[0]
    )

    # --------------------------------------------------------
    # LinkNet probabilities
    # --------------------------------------------------------

    linknet_probabilities = (
        outputs[
            "linknet_probabilities"
        ]
        .cpu()
        .numpy()[0]
    )

    # --------------------------------------------------------
    # Bi-LSTM probabilities
    # --------------------------------------------------------

    bilstm_probabilities = (
        outputs[
            "bilstm_probabilities"
        ]
        .cpu()
        .numpy()[0]
    )

    # --------------------------------------------------------
    # Convert class number to label
    # --------------------------------------------------------

    if prediction == 1:

        prediction_label = (
            "Defective"
        )

    else:

        prediction_label = (
            "Non-Defective"
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "prediction":
            prediction,

        "prediction_label":
            prediction_label,

        "confidence":
            confidence,

        "non_defective_probability":
            float(
                probabilities[0]
            ),

        "defective_probability":
            float(
                probabilities[1]
            ),

        "linknet_non_defective":
            float(
                linknet_probabilities[0]
            ),

        "linknet_defective":
            float(
                linknet_probabilities[1]
            ),

        "bilstm_non_defective":
            float(
                bilstm_probabilities[0]
            ),

        "bilstm_defective":
            float(
                bilstm_probabilities[1]
            )
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "SOFTWARE DEFECT PREDICTION TEST"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "\nERROR:"
        )

        print(
            "Trained model does not exist."
        )

        print(
            "\nRun:"
        )

        print(
            "    python train.py"
        )

        print(
            "\nfirst."
        )

        exit()

    # --------------------------------------------------------
    # Check feature file
    # --------------------------------------------------------

    if not os.path.exists(
        FEATURES_PATH
    ):

        print(
            "\nERROR:"
        )

        print(
            "Feature configuration does not exist."
        )

        print(
            "\nRun:"
        )

        print(
            "    python train.py"
        )

        print(
            "\nfirst."
        )

        exit()

    # --------------------------------------------------------
    # Load feature names
    # --------------------------------------------------------

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    print(
        "\nNumber of features:",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # Create example input
    #
    # This is only a test input.
    # The actual Streamlit application
    # will collect the values from the user.
    # --------------------------------------------------------

    example_input = {}

    for feature in feature_columns:

        example_input[
            feature
        ] = 0.0

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    result = predict(
        example_input
    )

    # --------------------------------------------------------
    # Display final result
    # --------------------------------------------------------

    print(
        "\nPrediction:",
        result[
            "prediction_label"
        ]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Final fusion
    # --------------------------------------------------------

    print(
        "\nFinal Score-Level Fusion:"
    )

    print(
        "Non-Defective:",
        f"{result['non_defective_probability'] * 100:.2f}%"
    )

    print(
        "Defective:",
        f"{result['defective_probability'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # LinkNet
    # --------------------------------------------------------

    print(
        "\nImproved LinkNet:"
    )

    print(
        "Non-Defective:",
        f"{result['linknet_non_defective'] * 100:.2f}%"
    )

    print(
        "Defective:",
        f"{result['linknet_defective'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Bi-LSTM
    # --------------------------------------------------------

    print(
        "\nBi-LSTM:"
    )

    print(
        "Non-Defective:",
        f"{result['bilstm_non_defective'] * 100:.2f}%"
    )

    print(
        "Defective:",
        f"{result['bilstm_defective'] * 100:.2f}%"
    )