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
# CONFIGURATION & HYPERPARAMETERS (SECTIONS 4.6 & 5)
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "defect_model.pth")
RESULTS_PATH = os.path.join(MODEL_DIR, "training_results.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42

# Score-level fusion weights (Equation 11): w1 + w2 = 1.0
LINKNET_WEIGHT = 0.50
BILSTM_WEIGHT = 0.50


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        fused_logits = outputs["fused_logits"]

        loss = criterion(fused_logits, y_batch)
        loss.backward()

        # Gradient clipping to stabilize Bi-LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        predictions = torch.argmax(fused_logits, dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            fused_logits = outputs["fused_logits"]
            fused_probs = outputs["fused_probability"]

            loss = criterion(fused_logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(fused_probs, dim=1)

            all_labels.extend(y_batch.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(fused_probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss, all_labels, all_predictions, all_probs


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }


def save_checkpoint(model, num_features, path=MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "linknet_weight": LINKNET_WEIGHT,
        "bilstm_weight": BILSTM_WEIGHT,
        "embedding_dim": 128
    }
    torch.save(checkpoint, path)
    print(f"Saved model checkpoint to: {path}")


def train_model():
    set_seed(RANDOM_STATE)
    device = get_device()
    print(f"\nTraining on device: {device}")

    # 1. Prepare Data
    X_train, X_test, y_train, y_test, feature_columns, scaler = prepare_data(
        test_size=0.20,
        random_state=RANDOM_STATE
    )
    num_features = len(feature_columns)

    # Class weighting for imbalance
    class_counts = np.bincount(y_train)
    weights = torch.tensor([1.0, class_counts[0] / max(class_counts[1], 1)], dtype=torch.float32).to(device)

    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE)

    # 2. Instantiate Hybrid Model (Improved LinkNet + Bi-LSTM + Score Fusion)
    model = HybridDefectModel(
        num_features=num_features,
        linknet_weight=LINKNET_WEIGHT,
        bilstm_weight=BILSTM_WEIGHT
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    print("\n" + "=" * 60)
    print("STARTING HYBRID LINKNET-BiLSTM MODEL TRAINING (30-35 EPOCHS)")
    print("=" * 60)

    best_f1 = -1.0
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_labels, te_preds, _ = evaluate(model, test_loader, criterion, device)
        te_acc = accuracy_score(te_labels, te_preds)
        te_f1 = f1_score(te_labels, te_preds, zero_division=0)

        scheduler.step()

        history["train_loss"].append(float(tr_loss))
        history["train_accuracy"].append(float(tr_acc))
        history["test_loss"].append(float(te_loss))
        history["test_accuracy"].append(float(te_acc))

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
                f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
                f"Test Loss: {te_loss:.4f}, Acc: {te_acc:.4f}, F1: {te_f1:.4f}"
            )

    # Final Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (COMPARED AGAINST PAPER BENCHMARK)")
    print("=" * 60)

    final_loss, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion, device)
    metrics = calculate_metrics(y_true, y_pred)

    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:  {metrics['f1_score'] * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Non-Defective", "Defective"], zero_division=0))
    print("Confusion Matrix:\n", np.array(metrics["confusion_matrix"]))

    # Save results & history
    save_checkpoint(model, num_features)

    training_results = {
        "dataset": "CM1 (PROMISE Repository)",
        "model_architecture": "Hybrid Improved LinkNet + Bi-LSTM with Score-Level Fusion",
        "num_features": num_features,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "linknet_weight": LINKNET_WEIGHT,
        "bilstm_weight": BILSTM_WEIGHT,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_matrix": metrics["confusion_matrix"],
        "feature_columns": feature_columns
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(training_results, f, indent=4)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Results written to {RESULTS_PATH} and {HISTORY_PATH}")
    return metrics, history


if __name__ == "__main__":
    train_model()