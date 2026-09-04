import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import prepare_data
from model import HybridDefectModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "defect_model.pth"
)

RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "training_results.json"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history.json"
)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

BATCH_SIZE = 16

EPOCHS = 105

LEARNING_RATE = 0.001

RANDOM_STATE = 42

LINKNET_WEIGHT = 0.5

BILSTM_WEIGHT = 0.5


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        "\nUsing device:",
        device
    )

    return device


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_data_loaders(
    X_train,
    X_test,
    y_train,
    y_test
):

    # --------------------------------------------------------
    # Convert training features to tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Convert labels to tensors
    # --------------------------------------------------------

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    # --------------------------------------------------------
    # Create data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        test_loader
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    # --------------------------------------------------------
    # Iterate through training batches
    # --------------------------------------------------------

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            X_batch
        )

        fused_logits = outputs[
            "fused_logits"
        ]

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            fused_logits,
            y_batch
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model parameters
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * X_batch.size(0)
        )

        predictions = torch.argmax(
            fused_logits,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(
    model,
    data_loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                X_batch
            )

            fused_logits = outputs[
                "fused_logits"
            ]

            fused_probability = outputs[
                "fused_probability"
            ]

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                fused_logits,
                y_batch
            )

            total_loss += (
                loss.item()
                * X_batch.size(0)
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            predictions = torch.argmax(
                fused_probability,
                dim=1
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_labels.extend(
                y_batch.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                fused_probability.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    all_probabilities = np.array(
        all_probabilities
    )

    average_loss = (
        total_loss
        /
        len(data_loader.dataset)
    )

    return (
        average_loss,
        all_labels,
        all_predictions,
        all_probabilities
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "confusion_matrix":
            matrix.tolist()
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    num_features
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "num_features":
            num_features,

        "linknet_weight":
            LINKNET_WEIGHT,

        "bilstm_weight":
            BILSTM_WEIGHT
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print(
        "\nModel saved to:",
        MODEL_PATH
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model():

    print(
        "=" * 60
    )

    print(
        "SOFTWARE DEFECT PREDICTION"
    )

    print(
        "HYBRID LINKNET + BI-LSTM"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    set_seed(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Prepare CM1 dataset
    # --------------------------------------------------------

    print(
        "\nPreparing CM1 dataset..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        scaler
    ) = prepare_data(
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Number of features
    # --------------------------------------------------------

    num_features = len(
        feature_columns
    )

    print(
        "\nNumber of features:",
        num_features
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        test_loader
    ) = create_data_loaders(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Create hybrid model
    # --------------------------------------------------------

    print(
        "\nCreating hybrid model..."
    )

    model = HybridDefectModel(

        num_features=num_features,

        linknet_weight=LINKNET_WEIGHT,

        bilstm_weight=BILSTM_WEIGHT
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(
    # weight=torch.tensor([1.0, 1.0], dtype=torch.float32, device = device )
# )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "test_loss": [],

        "test_accuracy": []
    }

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print(
        "\nStarting training..."
    )

    print(
        "=" * 60
    )

    for epoch in range(
        EPOCHS
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )
        )

        (
            test_loss,
            test_labels,
            test_predictions,
            test_probabilities
        ) = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        test_accuracy = accuracy_score(
            test_labels,
            test_predictions
        )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(
            float(train_loss)
        )

        history[
            "train_accuracy"
        ].append(
            float(train_accuracy)
        )

        history[
            "test_loss"
        ].append(
            float(test_loss)
        )

        history[
            "test_accuracy"
        ].append(
            float(test_accuracy)
        )

        # ----------------------------------------------------
        # Display epoch
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Test Loss: {test_loss:.4f} "
            f"| Test Acc: {test_accuracy:.4f}"
        )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL MODEL EVALUATION"
    )

    print(
        "=" * 60
    )

    (
        test_loss,
        y_true,
        y_pred,
        probabilities
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Display metrics
    # --------------------------------------------------------

    print(
        "\nTest Loss:",
        f"{test_loss:.4f}"
    )

    print(
        "Accuracy:",
        f"{metrics['accuracy']:.4f}"
    )

    print(
        "Precision:",
        f"{metrics['precision']:.4f}"
    )

    print(
        "Recall:",
        f"{metrics['recall']:.4f}"
    )

    print(
        "F1 Score:",
        f"{metrics['f1_score']:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Non-Defective",
                "Defective"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_model(
        model,
        num_features
    )

    # ========================================================
    # SAVE TRAINING RESULTS
    # ========================================================

    training_results = {

        "dataset":
            "CM1",

        "num_features":
            num_features,

        "train_samples":
            len(X_train),

        "test_samples":
            len(X_test),

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "linknet_weight":
            LINKNET_WEIGHT,

        "bilstm_weight":
            BILSTM_WEIGHT,

        "test_loss":
            float(test_loss),

        "accuracy":
            metrics[
                "accuracy"
            ],

        "precision":
            metrics[
                "precision"
            ],

        "recall":
            metrics[
                "recall"
            ],

        "f1_score":
            metrics[
                "f1_score"
            ],

        "confusion_matrix":
            metrics[
                "confusion_matrix"
            ],

        "feature_columns":
            feature_columns
    }

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w"
    ) as file:

        json.dump(
            training_results,
            file,
            indent=4
        )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    print(
        "\nTraining results saved to:",
        RESULTS_PATH
    )

    print(
        "Training history saved to:",
        HISTORY_PATH
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  - models/defect_model.pth"
    )

    print(
        "  - models/scaler.pkl"
    )

    print(
        "  - models/features.pkl"
    )

    print(
        "  - models/training_results.json"
    )

    print(
        "  - models/training_history.json"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_model()