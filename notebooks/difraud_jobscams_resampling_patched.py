#!/usr/bin/env python3
"""
DiFrauD Job Scams resampling experiment: patched, Colab-ready version.

This version avoids the Hugging Face dataset loading script difraud.py because
recent versions of the Hugging Face `datasets` package no longer support
script-based datasets cleanly, and some older versions fail because
`datasets.tasks.TextClassification` has been removed.

Instead, this script loads the Job Scams JSONL files directly from the
Hugging Face repository.

What this script does
---------------------
1. Downloads the Job Scams train/validation/test JSONL files directly.
2. Uses the official train split for training and the official test split for
   evaluation.
3. Extracts TF-IDF features from job-posting text.
4. Applies TruncatedSVD to obtain dense vectors.
5. Compares:
   - Baseline: no resampling
   - SMOTE: oversampling
   - Random undersampling: undersampling
   - SMOTE + ENN: hybrid oversampling + cleaning
6. Trains the same logistic-regression classifier for every condition.
7. Saves chapter-ready figures:
   - class distribution
   - metrics comparison with values above bars
   - confusion matrices
   - optional 2D representation-space visualization
8. Saves the numerical results as CSV and JSON metadata.

Install in Colab
----------------
!pip install imbalanced-learn scikit-learn pandas numpy matplotlib

Optional:
!pip install umap-learn

Run:
!python difraud_jobscams_resampling_patched.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from sklearn.decomposition import TruncatedSVD, PCA
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


RANDOM_STATE = 42
OUTDIR = Path("difraud_jobscams_resampling_outputs")
OUTDIR.mkdir(exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/difraud/difraud/resolve/main/job_scams"
URLS = {
    "train": f"{HF_BASE}/train.jsonl",
    "validation": f"{HF_BASE}/validation.jsonl",
    "test": f"{HF_BASE}/test.jsonl",
}


def load_jsonl_from_url(url: str) -> pd.DataFrame:
    """Load a JSONL file from a URL into a pandas DataFrame."""
    return pd.read_json(url, lines=True)


def load_job_scams() -> Dict[str, pd.DataFrame]:
    """Load the Job Scams component of DiFrauD directly from JSONL files."""
    data = {}
    for split, url in URLS.items():
        print(f"Loading {split} from {url}")
        data[split] = load_jsonl_from_url(url)
    return data


def get_xy(df: pd.DataFrame) -> Tuple[list[str], np.ndarray]:
    """Extract texts and binary labels from a Job Scams split."""
    if "text" not in df.columns:
        raise ValueError(f"Expected a 'text' column, found columns: {list(df.columns)}")
    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column, found columns: {list(df.columns)}")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()
    return texts, labels


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary classification metrics for the positive scam class."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "PR-AUC": average_precision_score(y_true, y_score),
    }

    try:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["ROC-AUC"] = np.nan

    return metrics


def train_eval_one(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sampler=None,
):
    """Train and evaluate one resampling condition."""
    if sampler is None:
        X_res, y_res = X_train, y_train
    else:
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    clf = LogisticRegression(
        max_iter=3000,
        class_weight=None,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_res, y_res)

    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_score, y_pred)
    metrics["Method"] = name
    metrics["Train benign after resampling"] = int(np.sum(y_res == 0))
    metrics["Train scam after resampling"] = int(np.sum(y_res == 1))

    return metrics, clf, X_res, y_res, y_pred, y_score


def plot_class_distributions(class_rows: list[dict]) -> Path:
    """Save a book-ready class-distribution plot."""
    df = pd.DataFrame(class_rows)
    methods = df["Method"].tolist()
    benign = df["Benign"].to_numpy()
    scam = df["Scam"].to_numpy()

    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - width / 2, benign, width, label="Benign")
    ax.bar(x + width / 2, scam, width, label="Scam")

    max_y = max(max(benign), max(scam))
    for i, v in enumerate(benign):
        ax.text(
            i - width / 2,
            v + max_y * 0.015,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    for i, v in enumerate(scam):
        ax.text(
            i + width / 2,
            v + max_y * 0.015,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.set_title("Job Scams Training Distribution Before and After Resampling", fontsize=16, pad=12)
    ax.set_xlabel("Training condition", fontsize=13)
    ax.set_ylabel("Number of training examples", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fontsize=10,
    )

    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_class_distribution_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metrics(results_df: pd.DataFrame) -> Path:
    """Save a book-ready grouped metrics plot with values above the bars."""
    metric_cols = ["Precision", "Recall", "F1", "Balanced accuracy", "MCC", "PR-AUC"]
    plot_df = results_df.set_index("Method")[metric_cols]

    methods = plot_df.index.tolist()
    metrics = plot_df.columns.tolist()
    values = plot_df.to_numpy()

    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_methods)
    width = 0.12

    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for i, metric in enumerate(metrics):
        offsets = x + (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(offsets, values[:, i], width=width, label=metric)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("Job Scams Test Performance by Resampling Method", fontsize=18, pad=14)
    ax.set_xlabel("Method", fontsize=15, labelpad=10)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, fontsize=13)
    ax.set_ylim(0, 1.05)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=11,
    )

    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_metrics_comparison_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusions(y_test: np.ndarray, predictions: dict[str, np.ndarray]) -> Path:
    """Save confusion matrices for all methods."""
    methods = list(predictions.keys())

    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 3.8))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        cm = confusion_matrix(y_test, predictions[method], labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Scam"],
        )
        disp.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(method, fontsize=11)

    fig.suptitle("Job Scams Test Confusion Matrices", y=1.04, fontsize=15)
    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_confusion_matrices_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def reduce_to_2d(X: np.ndarray, y: np.ndarray, max_points: int = 2500):
    """Reduce dense vectors to two dimensions for visualization."""
    rng = np.random.default_rng(RANDOM_STATE)

    if len(y) > max_points:
        idx = rng.choice(len(y), size=max_points, replace=False)
        X_small = X[idx]
        y_small = y[idx]
    else:
        X_small = X
        y_small = y

    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.15,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        Z = reducer.fit_transform(X_small)
        method = "UMAP"
    except Exception:
        reducer = PCA(n_components=2, random_state=RANDOM_STATE)
        Z = reducer.fit_transform(X_small)
        method = "PCA"

    return Z, y_small, method


def plot_2d_views(resampled_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Save a qualitative 2D visualization of the resampled training sets."""
    methods = list(resampled_data.keys())

    fig, axes = plt.subplots(1, len(methods), figsize=(4.1 * len(methods), 3.8))
    if len(methods) == 1:
        axes = [axes]

    reduction_method = None

    for ax, method_name in zip(axes, methods):
        X_res, y_res = resampled_data[method_name]
        Z, y_small, red = reduce_to_2d(X_res, y_res)
        reduction_method = red

        ax.scatter(
            Z[y_small == 0, 0],
            Z[y_small == 0, 1],
            s=5,
            alpha=0.45,
            label="Benign",
        )
        ax.scatter(
            Z[y_small == 1, 0],
            Z[y_small == 1, 1],
            s=8,
            alpha=0.65,
            label="Scam",
        )
        ax.set_title(method_name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"2D View of Original and Resampled Training Sets ({reduction_method})", y=1.03, fontsize=14)
    fig.tight_layout()
    path = OUTDIR / "fig_jobscams_2d_resampling_views_book.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    warnings.filterwarnings("ignore")

    data = load_job_scams()
    train_texts, y_train = get_xy(data["train"])
    val_texts, y_val = get_xy(data["validation"])
    test_texts, y_test = get_xy(data["test"])

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

    # Feature extraction.
    # Fit only on the training split to avoid test-set leakage.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        min_df=3,
        max_df=0.95,
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    n_components = min(200, X_train_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    scaler = StandardScaler()

    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    X_train = scaler.fit_transform(X_train_svd)
    X_test = scaler.transform(X_test_svd)

    # Robust SMOTE neighbor setting.
    minority_count = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    k_neighbors = max(1, min(5, minority_count - 1))

    methods = {
        "Baseline": None,
        "SMOTE": SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors),
        "Random undersampling": RandomUnderSampler(random_state=RANDOM_STATE),
        "SMOTE + ENN": SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors),
        ),
    }

    results = []
    predictions = {}
    resampled_data = {"Original": (X_train, y_train)}
    class_rows = [
        {
            "Method": "Original",
            "Benign": int(np.sum(y_train == 0)),
            "Scam": int(np.sum(y_train == 1)),
        }
    ]

    for name, sampler in methods.items():
        print(f"\nRunning: {name}")
        metrics, clf, X_res, y_res, y_pred, y_score = train_eval_one(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            sampler,
        )

        results.append(metrics)
        predictions[name] = y_pred

        print(
            f"  precision={metrics['Precision']:.3f}, "
            f"recall={metrics['Recall']:.3f}, "
            f"F1={metrics['F1']:.3f}, "
            f"balanced_accuracy={metrics['Balanced accuracy']:.3f}, "
            f"MCC={metrics['MCC']:.3f}, "
            f"PR-AUC={metrics['PR-AUC']:.3f}"
        )

        if name != "Baseline":
            resampled_data[name] = (X_res, y_res)
            class_rows.append(
                {
                    "Method": name,
                    "Benign": int(np.sum(y_res == 0)),
                    "Scam": int(np.sum(y_res == 1)),
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "Method",
            "Accuracy",
            "Balanced accuracy",
            "Precision",
            "Recall",
            "F1",
            "MCC",
            "PR-AUC",
            "ROC-AUC",
            "Train benign after resampling",
            "Train scam after resampling",
        ]
    ]

    # Save machine-readable results. We intentionally do not generate a LaTeX
    # table by default because the metrics figure places the values above bars.
    results_df.to_csv(OUTDIR / "jobscams_resampling_results.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUTDIR / "jobscams_resampling_class_counts.csv", index=False)

    figure_paths = {
        "class_distribution": str(plot_class_distributions(class_rows)),
        "metrics_comparison": str(plot_metrics(results_df)),
        "confusion_matrices": str(plot_confusions(y_test, predictions)),
        "two_d_views": str(plot_2d_views(resampled_data)),
    }

    metadata = {
        "dataset": "DiFrauD Job Scams",
        "source": URLS,
        "train_size": int(len(y_train)),
        "validation_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "train_class_counts": {
            "benign_0": int(np.sum(y_train == 0)),
            "scam_1": int(np.sum(y_train == 1)),
        },
        "test_class_counts": {
            "benign_0": int(np.sum(y_test == 0)),
            "scam_1": int(np.sum(y_test == 1)),
        },
        "tfidf_features": int(X_train_tfidf.shape[1]),
        "svd_components": int(n_components),
        "random_state": RANDOM_STATE,
        "figure_paths": figure_paths,
    }

    with open(OUTDIR / "jobscams_resampling_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== DiFrauD Job Scams resampling experiment complete ===")
    print(json.dumps(metadata, indent=2))

    print("\nResults:")
    print(results_df.round(3).to_string(index=False))

    print(f"\nOutputs written to: {OUTDIR.resolve()}")
    print("\nRecommended chapter figures:")
    print(f"  {figure_paths['class_distribution']}")
    print(f"  {figure_paths['metrics_comparison']}")
    print(f"  {figure_paths['confusion_matrices']}")


if __name__ == "__main__":
    main()
