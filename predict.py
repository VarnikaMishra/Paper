import os
import joblib
import torch
import numpy as np
import pandas as pd

from model import HybridDefectModel
from preprocess import preprocess_new_input, preprocess_code_snippet
from code2vec_extractor import extractor


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join("models", "defect_model.pth")
FEATURES_PATH = os.path.join("models", "features.pkl")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run train.py first.")

    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_features = checkpoint["num_features"]
    linknet_weight = checkpoint.get("linknet_weight", 0.5)
    bilstm_weight = checkpoint.get("bilstm_weight", 0.5)

    model = HybridDefectModel(
        num_features=num_features,
        linknet_weight=linknet_weight,
        bilstm_weight=bilstm_weight
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device, num_features


def predict_metrics(input_data):
    """
    Predicts defect probability for tabular software metrics.
    """
    model, device, num_features = load_model()
    processed_input = preprocess_new_input(input_data)
    X_tensor = torch.tensor(processed_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model.predict(X_tensor)

    prediction = int(outputs["prediction"].cpu().numpy()[0])
    confidence = float(outputs["confidence"].cpu().numpy()[0])
    probs = outputs["probabilities"].cpu().numpy()[0]
    linknet_probs = outputs["linknet_probabilities"].cpu().numpy()[0]
    bilstm_probs = outputs["bilstm_probabilities"].cpu().numpy()[0]

    return {
        "input_type": "tabular_metrics",
        "prediction": prediction,
        "prediction_label": "Defective" if prediction == 1 else "Non-Defective",
        "confidence": confidence,
        "non_defective_probability": float(probs[0]),
        "defective_probability": float(probs[1]),
        "linknet_non_defective": float(linknet_probs[0]),
        "linknet_defective": float(linknet_probs[1]),
        "bilstm_non_defective": float(bilstm_probs[0]),
        "bilstm_defective": float(bilstm_probs[1])
    }


def predict_code(code_str):
    """
    Predicts defect probability directly from raw code using Code2Vec AST Feature Extraction (Section 4.3).
    """
    model, device, num_features = load_model()
    code_vec, contexts = preprocess_code_snippet(code_str)

    # Project or interpolate code vector to match model input dimension
    if len(code_vec) != num_features:
        # Interpolate 128-dim Code2Vec vector to num_features
        vec_tensor = torch.tensor(code_vec, dtype=torch.float32).view(1, 1, -1)
        adapted_vec = torch.nn.functional.interpolate(vec_tensor, size=num_features, mode='linear', align_corners=False).squeeze(0)
    else:
        adapted_vec = torch.tensor(code_vec, dtype=torch.float32).unsqueeze(0)

    adapted_vec = adapted_vec.to(device)

    with torch.no_grad():
        outputs = model.predict(adapted_vec)

    prediction = int(outputs["prediction"].cpu().numpy()[0])
    confidence = float(outputs["confidence"].cpu().numpy()[0])
    probs = outputs["probabilities"].cpu().numpy()[0]
    linknet_probs = outputs["linknet_probabilities"].cpu().numpy()[0]
    bilstm_probs = outputs["bilstm_probabilities"].cpu().numpy()[0]

    return {
        "input_type": "source_code",
        "prediction": prediction,
        "prediction_label": "Defective" if prediction == 1 else "Non-Defective",
        "confidence": confidence,
        "non_defective_probability": float(probs[0]),
        "defective_probability": float(probs[1]),
        "linknet_non_defective": float(linknet_probs[0]),
        "linknet_defective": float(linknet_probs[1]),
        "bilstm_non_defective": float(bilstm_probs[0]),
        "bilstm_defective": float(bilstm_probs[1]),
        "ast_contexts_count": len(contexts),
        "sample_ast_paths": contexts[:5]
    }


def predict(input_data):
    """
    Polymorphic prediction interface supporting:
    - string (source code snippet)
    - dict or DataFrame (software metrics)
    """
    if isinstance(input_data, str):
        return predict_code(input_data)
    else:
        return predict_metrics(input_data)


if __name__ == "__main__":
    # Test Code2Vec prediction on a code snippet
    sample_code = """
    def compute_average(values):
        total = 0
        for val in values:
            total += val
        return total / len(values) if len(values) > 0 else 0
    """
    print("Testing Code2Vec raw code prediction:")
    res = predict(sample_code)
    print(res)