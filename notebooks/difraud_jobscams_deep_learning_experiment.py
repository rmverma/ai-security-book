#!/usr/bin/env python3
"""
DiFrauD Job Scams deep-learning experiment: Colab-ready version.

This script is intended for a Deep Learning chapter case study. It compares:
1. TF-IDF + Logistic Regression baseline
2. TF-IDF + SVD + MLP
3. CNN text classifier
4. BiLSTM text classifier
5. DistilBERT Transformer classifier, optional but enabled by default

The script loads the raw Job Scams JSONL files directly from Hugging Face.
This avoids the known issue with the DiFrauD dataset loader script `difraud.py`.

Recommended Colab runtime:
    Runtime -> Change runtime type -> T4 GPU

Install:
    !pip install -q pandas numpy matplotlib scikit-learn tensorflow torch transformers tqdm

Run:
    !python difraud_jobscams_deep_learning_experiment.py

If the Transformer model is too slow, set RUN_TRANSFORMER = False below.
"""

from __future__ import annotations

import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# ----------------------------
# Configuration
# ----------------------------

RANDOM_STATE = 42
OUTDIR = Path("difraud_jobscams_deep_learning_outputs")
OUTDIR.mkdir(exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/difraud/difraud/resolve/main/job_scams"
URLS = {
    "train": f"{HF_BASE}/train.jsonl",
    "validation": f"{HF_BASE}/validation.jsonl",
    "test": f"{HF_BASE}/test.jsonl",
}

# Keras text models
MAX_TOKENS = 30000
SEQUENCE_LENGTH = 256
EMBED_DIM = 128
BATCH_SIZE = 64
KERAS_EPOCHS = 8
PATIENCE = 2

# SVD representation for MLP and Logistic Regression
TFIDF_MAX_FEATURES = 30000
SVD_COMPONENTS = 200

# Transformer model
RUN_TRANSFORMER = True
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MAX_LENGTH = 192
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 2
TRANSFORMER_LR = 2e-5

# For fast debugging only. Leave as None for the actual experiment.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None


def set_seeds(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seeds()


def load_jsonl_from_url(url: str) -> pd.DataFrame:
    return pd.read_json(url, lines=True)


def load_job_scams() -> Dict[str, pd.DataFrame]:
    data = {}
    for split, url in URLS.items():
        print(f"Loading {split} from {url}")
        data[split] = load_jsonl_from_url(url)
    return data


def maybe_subsample(texts: List[str], labels: np.ndarray, max_samples):
    if max_samples is None or len(labels) <= max_samples:
        return texts, labels
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(labels), size=max_samples, replace=False)
    idx = np.sort(idx)
    return [texts[i] for i in idx], labels[idx]


def get_xy(df: pd.DataFrame, max_samples=None) -> Tuple[List[str], np.ndarray]:
    if "text" not in df.columns:
        raise ValueError(f"Expected a 'text' column. Found: {list(df.columns)}")
    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column. Found: {list(df.columns)}")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()
    return maybe_subsample(texts, labels, max_samples)


def select_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Select a threshold on validation data that maximizes positive-class F1."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_score >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    out = {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "PR-AUC": average_precision_score(y_true, y_score),
    }
    try:
        out["ROC-AUC"] = roc_auc_score(y_true, y_score)
    except ValueError:
        out["ROC-AUC"] = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["TN"] = int(cm[0, 0])
    out["FP"] = int(cm[0, 1])
    out["FN"] = int(cm[1, 0])
    out["TP"] = int(cm[1, 1])
    return out


def evaluate_from_scores(model_name: str, y_val: np.ndarray, val_score: np.ndarray,
                         y_test: np.ndarray, test_score: np.ndarray):
    threshold = select_threshold_by_f1(y_val, val_score)
    metrics = compute_metrics(y_test, test_score, threshold)
    metrics["Model"] = model_name
    y_pred = (test_score >= threshold).astype(int)
    return metrics, y_pred


def make_svd_features(train_texts: List[str], val_texts: List[str], test_texts: List[str]):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        min_df=3,
        max_df=0.95,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_val_tfidf = vectorizer.transform(val_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    n_components = min(SVD_COMPONENTS, X_train_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_val_svd = svd.transform(X_val_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_svd)
    X_val = scaler.transform(X_val_svd)
    X_test = scaler.transform(X_test_svd)
    return X_train, X_val, X_test, vectorizer, svd


def run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nRunning TF-IDF + SVD + Logistic Regression baseline")
    clf = LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    val_score = clf.predict_proba(X_val)[:, 1]
    test_score = clf.predict_proba(X_test)[:, 1]
    return evaluate_from_scores("TF-IDF + LR", y_val, val_score, y_test, test_score)


def build_mlp(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def run_mlp(X_train, y_train, X_val, y_val, X_test, y_test, class_weight_dict):
    print("\nRunning TF-IDF + SVD + MLP")
    model = build_mlp(X_train.shape[1])
    early = callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=KERAS_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early],
        verbose=2,
    )
    val_score = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    test_score = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    metrics, y_pred = evaluate_from_scores("MLP", y_val, val_score, y_test, test_score)
    return metrics, y_pred, history


def make_text_vectorizer(train_texts: List[str]) -> tf.keras.layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )
    text_ds = tf.data.Dataset.from_tensor_slices(train_texts).batch(128)
    vectorizer.adapt(text_ds)
    return vectorizer


def text_to_sequences(vectorizer, texts: List[str]) -> np.ndarray:
    return vectorizer(np.array(texts)).numpy()


def build_cnn_classifier() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH,)),
        layers.Embedding(MAX_TOKENS, EMBED_DIM),
        layers.Conv1D(128, kernel_size=5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.35),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def build_bilstm_classifier() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH,)),
        layers.Embedding(MAX_TOKENS, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.35),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def run_keras_sequence_model(model_name: str, model: tf.keras.Model,
                             X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test,
                             class_weight_dict):
    print(f"\nRunning {model_name}")
    early = callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=KERAS_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early],
        verbose=2,
    )
    val_score = model.predict(X_val_seq, batch_size=BATCH_SIZE, verbose=0).ravel()
    test_score = model.predict(X_test_seq, batch_size=BATCH_SIZE, verbose=0).ravel()
    metrics, y_pred = evaluate_from_scores(model_name, y_val, val_score, y_test, test_score)
    return metrics, y_pred, history


def run_transformer(train_texts, y_train, val_texts, y_val, test_texts, y_test, class_weight_dict):
    print(f"\nRunning Transformer: {TRANSFORMER_MODEL_NAME}")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        from tqdm.auto import tqdm
    except Exception as e:
        print("Transformer dependencies are unavailable. Skipping Transformer.")
        print(f"Reason: {e}")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Transformer device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=TRANSFORMER_MAX_LENGTH,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in item.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return item

    train_loader = DataLoader(TextDataset(train_texts, y_train), batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_texts, y_val), batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TextDataset(test_texts, y_test), batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_NAME, num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRANSFORMER_LR)
    total_steps = len(train_loader) * TRANSFORMER_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([float(class_weight_dict[0]), float(class_weight_dict[1])], dtype=torch.float32).to(device)
    )

    for epoch in range(TRANSFORMER_EPOCHS):
        model.train()
        print(f"Transformer epoch {epoch + 1}/{TRANSFORMER_EPOCHS}")
        total_loss = 0.0
        for batch in tqdm(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        print(f"  mean train loss: {total_loss / max(1, len(train_loader)):.4f}")

    def predict_scores(loader):
        model.eval()
        all_scores = []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_scores.append(probs.detach().cpu().numpy())
        return np.concatenate(all_scores)

    val_score = predict_scores(val_loader)
    test_score = predict_scores(test_loader)
    metrics, y_pred = evaluate_from_scores("DistilBERT", y_val, val_score, y_test, test_score)
    return metrics, y_pred


def plot_metrics(results_df: pd.DataFrame) -> Path:
    metric_cols = ["Precision", "Recall", "F1", "Balanced accuracy", "MCC", "PR-AUC"]
    plot_df = results_df.set_index("Model")[metric_cols]
    methods = plot_df.index.tolist()
    metrics = plot_df.columns.tolist()
    values = plot_df.to_numpy()
    x = np.arange(len(methods))
    width = 0.12
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    for i, metric in enumerate(metrics):
        offsets = x + (i - (len(metrics) - 1) / 2) * width
        bars = ax.bar(offsets, values[:, i], width=width, label=metric)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title("Job Scams Test Performance by Deep Learning Model", fontsize=18, pad=14)
    ax.set_xlabel("Model", fontsize=15, labelpad=10)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=True, fontsize=10)
    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_dl_metrics_comparison_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall_f1(results_df: pd.DataFrame) -> Path:
    metric_cols = ["Precision", "Recall", "F1"]
    plot_df = results_df.set_index("Model")[metric_cols]
    methods = plot_df.index.tolist()
    metrics = plot_df.columns.tolist()
    values = plot_df.to_numpy()
    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for i, metric in enumerate(metrics):
        offsets = x + (i - 1) * width
        bars = ax.bar(offsets, values[:, i], width=width, label=metric)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("Precision--Recall Tradeoff on Job Scams", fontsize=17, pad=12)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True, fontsize=10)
    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_dl_precision_recall_f1_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusions(y_test: np.ndarray, predictions: Dict[str, np.ndarray]) -> Path:
    methods = list(predictions.keys())
    cols = min(3, len(methods))
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, method in zip(axes, methods):
        cm = confusion_matrix(y_test, predictions[method], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Scam"])
        disp.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(method, fontsize=11)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle("Job Scams Test Confusion Matrices for Deep Learning Models", y=1.02, fontsize=15)
    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_dl_confusion_matrices_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_histories(histories: Dict[str, tf.keras.callbacks.History]) -> Path | None:
    if not histories:
        return None
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for name, hist in histories.items():
        if "val_pr_auc" in hist.history:
            ax.plot(hist.history["val_pr_auc"], marker="o", label=f"{name} validation PR-AUC")
    ax.set_title("Validation PR-AUC During Training", fontsize=16, pad=12)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation PR-AUC", fontsize=13)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_dl_training_histories_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    warnings.filterwarnings("ignore")
    print("\n=== DiFrauD Job Scams Deep Learning Experiment ===")
    print(f"TensorFlow version: {tf.__version__}")
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    data = load_job_scams()
    train_texts, y_train = get_xy(data["train"], MAX_TRAIN_SAMPLES)
    val_texts, y_val = get_xy(data["validation"], MAX_VAL_SAMPLES)
    test_texts, y_test = get_xy(data["test"], MAX_TEST_SAMPLES)

    print("\nSplit sizes:")
    print(f"  train:      {len(y_train):,}")
    print(f"  validation: {len(y_val):,}")
    print(f"  test:       {len(y_test):,}")
    print("\nTraining class counts:")
    print(f"  benign / 0: {int(np.sum(y_train == 0)):,}")
    print(f"  scam   / 1: {int(np.sum(y_train == 1)):,}")
    print("\nTest class counts:")
    print(f"  benign / 0: {int(np.sum(y_test == 0)):,}")
    print(f"  scam   / 1: {int(np.sum(y_test == 1)):,}")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}
    print("\nClass weights:", class_weight_dict)

    results = []
    predictions = {}
    histories = {}
    start = time.time()

    X_train_svd, X_val_svd, X_test_svd, _, _ = make_svd_features(train_texts, val_texts, test_texts)

    metrics, y_pred = run_logistic_regression(X_train_svd, y_train, X_val_svd, y_val, X_test_svd, y_test)
    results.append(metrics)
    predictions[metrics["Model"]] = y_pred

    metrics, y_pred, hist = run_mlp(X_train_svd, y_train, X_val_svd, y_val, X_test_svd, y_test, class_weight_dict)
    results.append(metrics)
    predictions[metrics["Model"]] = y_pred
    histories[metrics["Model"]] = hist

    text_vectorizer = make_text_vectorizer(train_texts)
    X_train_seq = text_to_sequences(text_vectorizer, train_texts)
    X_val_seq = text_to_sequences(text_vectorizer, val_texts)
    X_test_seq = text_to_sequences(text_vectorizer, test_texts)

    metrics, y_pred, hist = run_keras_sequence_model(
        "CNN", build_cnn_classifier(), X_train_seq, y_train, X_val_seq, y_val,
        X_test_seq, y_test, class_weight_dict
    )
    results.append(metrics)
    predictions[metrics["Model"]] = y_pred
    histories[metrics["Model"]] = hist

    metrics, y_pred, hist = run_keras_sequence_model(
        "BiLSTM", build_bilstm_classifier(), X_train_seq, y_train, X_val_seq, y_val,
        X_test_seq, y_test, class_weight_dict
    )
    results.append(metrics)
    predictions[metrics["Model"]] = y_pred
    histories[metrics["Model"]] = hist

    if RUN_TRANSFORMER:
        metrics, y_pred = run_transformer(train_texts, y_train, val_texts, y_val, test_texts, y_test, class_weight_dict)
        if metrics is not None:
            results.append(metrics)
            predictions[metrics["Model"]] = y_pred

    results_df = pd.DataFrame(results)
    ordered_cols = [
        "Model", "Threshold", "Accuracy", "Balanced accuracy", "Precision", "Recall",
        "F1", "MCC", "PR-AUC", "ROC-AUC", "TN", "FP", "FN", "TP"
    ]
    results_df = results_df[ordered_cols]
    results_path = OUTDIR / "jobscams_deep_learning_results.csv"
    results_df.to_csv(results_path, index=False)

    figure_paths = {
        "metrics": str(plot_metrics(results_df)),
        "precision_recall_f1": str(plot_precision_recall_f1(results_df)),
        "confusion_matrices": str(plot_confusions(y_test, predictions)),
    }
    hist_path = plot_training_histories(histories)
    if hist_path is not None:
        figure_paths["training_histories"] = str(hist_path)

    metadata = {
        "dataset": "DiFrauD Job Scams",
        "source": URLS,
        "train_size": int(len(y_train)),
        "validation_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "train_class_counts": {"benign_0": int(np.sum(y_train == 0)), "scam_1": int(np.sum(y_train == 1))},
        "validation_class_counts": {"benign_0": int(np.sum(y_val == 0)), "scam_1": int(np.sum(y_val == 1))},
        "test_class_counts": {"benign_0": int(np.sum(y_test == 0)), "scam_1": int(np.sum(y_test == 1))},
        "class_weight": class_weight_dict,
        "threshold_selection": "Threshold selected on validation split to maximize positive-class F1.",
        "random_state": RANDOM_STATE,
        "max_tokens": MAX_TOKENS,
        "sequence_length": SEQUENCE_LENGTH,
        "run_transformer": RUN_TRANSFORMER,
        "transformer_model": TRANSFORMER_MODEL_NAME if RUN_TRANSFORMER else None,
        "elapsed_seconds": float(time.time() - start),
        "figure_paths": figure_paths,
    }
    with open(OUTDIR / "jobscams_deep_learning_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Experiment complete ===")
    print("\nResults:")
    print(results_df.round(3).to_string(index=False))
    print("\nSaved outputs:")
    print(f"  {results_path}")
    for k, v in figure_paths.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs are in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
