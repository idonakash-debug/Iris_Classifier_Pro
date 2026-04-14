# -*- coding: utf-8 -*-
"""
Iris_Classifier_Pro - src/classifier.py  (v2 - Improved)
==========================================================
שיפורים מיושמים:
  1. StandardScaler   - נרמול z-score של המאפיינים
  2. פיצול 2D         - קטגוריה 4 לפי petal_length AND petal_width
  3. MLP + ReLU       - שכבה נסתרת עם 16 נוירונים
  4. Cross-Entropy    - פונקציית שגיאה מתאימה לסיווג (+ MSE למעקב)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# --- הגדרות גלובליות -------------------------------------------------------
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
LEARNING_RATE = 0.1
EPOCHS        = 2000
NUM_CLASSES   = 4
HIDDEN_SIZE   = 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
DOCS_DIR = os.path.join(BASE_DIR, '..', 'docs')

CLASS_NAMES = ['setosa', 'versicolor', 'virginica-small', 'virginica-large']


# --- 1. טעינה והכנת נתונים -------------------------------------------------

def load_and_prepare_data():
    """
    טוען Iris Dataset ויוצר קטגוריה 4 בפיצול דו-מימדי.

    שיפור 2 - פיצול 2D:
      virginica-large  אם  petal_length > median_pl  AND  petal_width > median_pw
      virginica-small  בכל שאר המקרים
    כך הגבול בין הקטגוריות חד יותר ואחוז החפיפה קטן.
    """
    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target.copy()

    virginica_idx = np.where(y == 2)[0]
    median_pl = np.median(X[virginica_idx, 2])
    median_pw = np.median(X[virginica_idx, 3])

    for i in virginica_idx:
        if X[i, 2] > median_pl and X[i, 3] > median_pw:
            y[i] = 3

    return X, y


def standard_scale(X_train, X_test):
    """
    שיפור 1 - נרמול z-score.
    הסטטיסטיקות מחושבות על Train בלבד ומוחלות על שניהם למניעת data leakage.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def save_datasets(X, y, X_train, X_test, y_train, y_test):
    os.makedirs(DATA_DIR, exist_ok=True)
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
    for fname, (features, labels) in {
        'iris_raw.csv':   (X,       y),
        'iris_train.csv': (X_train, y_train),
        'iris_test.csv':  (X_test,  y_test),
    }.items():
        pd.DataFrame(np.column_stack([features, labels]), columns=cols).to_csv(
            os.path.join(DATA_DIR, fname), index=False)
    print(f"[Data] Saved CSVs -> data/  (raw={len(X)}, train={len(X_train)}, test={len(X_test)})")


# --- 2. עזרי מודל ----------------------------------------------------------

def one_hot_encode(y, num_classes=NUM_CLASSES):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


def softmax(Z):
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z   = np.exp(shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def relu(Z):
    return np.maximum(0, Z)


def relu_grad(Z):
    return (Z > 0).astype(float)


def compute_cross_entropy(Y_true, Y_pred):
    """
    שיפור 4 - Cross-Entropy Loss.
    E_CE = -1/N * sum(yi * log(yi_tilde))
    גרדיאנטים חזקים יותר כשהמודל בטוח בטעות.
    """
    eps = 1e-12
    return float(-np.mean(np.sum(Y_true * np.log(Y_pred + eps), axis=1)))


def compute_mse(Y_true, Y_pred):
    """E_MSE = 1/N * sum((yi - yi_tilde)^2) - נשמר לגרף ולדוח."""
    return float(np.mean((Y_true - Y_pred) ** 2))


# --- 3. אימון MLP (Gradient Descent) ---------------------------------------

def train(X_train, Y_train, learning_rate=LEARNING_RATE, epochs=EPOCHS):
    """
    שיפור 3 - MLP עם שכבה נסתרת + ReLU.
    ארכיטקטורה: Input(4) -> Hidden(16, ReLU) -> Output(4, Softmax)

    שיפור 4 - Cross-Entropy Loss עם Backpropagation מלא.

    החזרה:
        weights    tuple  (W1, b1, W2, b2)
        ce_history list   Cross-Entropy בכל epoch
        mse_history list  MSE בכל epoch
    """
    N, F = X_train.shape
    C, H = Y_train.shape[1], HIDDEN_SIZE

    np.random.seed(RANDOM_STATE)
    # Xavier initialization - מניעת פיצוץ/כיבוי גרדיאנטים
    W1 = np.random.randn(F, H) * np.sqrt(2.0 / F)
    b1 = np.zeros((1, H))
    W2 = np.random.randn(H, C) * np.sqrt(2.0 / H)
    b2 = np.zeros((1, C))

    ce_history  = []
    mse_history = []

    for epoch in range(epochs):
        # -- Forward pass --
        Z1     = X_train @ W1 + b1    # (N, H)
        A1     = relu(Z1)             # (N, H)
        Z2     = A1 @ W2 + b2         # (N, C)
        Y_pred = softmax(Z2)          # (N, C)

        ce_history.append(compute_cross_entropy(Y_train, Y_pred))
        mse_history.append(compute_mse(Y_train, Y_pred))

        # -- Backward pass --
        # שכבה 2: נגזרת CE + Softmax = Y_pred - Y_true
        dZ2 = (Y_pred - Y_train) / N  # (N, C)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # שכבה 1: נגזרת ReLU
        dA1 = dZ2 @ W2.T              # (N, H)
        dZ1 = dA1 * relu_grad(Z1)     # (N, H)
        dW1 = X_train.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # -- עדכון משקולות --
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if (epoch + 1) % 400 == 0:
            print(f"  Epoch {epoch+1:>4}/{epochs}  |  CE = {ce_history[-1]:.5f}"
                  f"  |  MSE = {mse_history[-1]:.5f}")

    return (W1, b1, W2, b2), ce_history, mse_history


# --- 4. חיזוי והערכה -------------------------------------------------------

def predict(X, weights):
    W1, b1, W2, b2 = weights
    A1 = relu(X @ W1 + b1)
    return np.argmax(softmax(A1 @ W2 + b2), axis=1)


def compute_accuracy(y_true, y_pred):
    return float(np.mean(y_true.astype(int) == y_pred.astype(int)))


def build_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


# --- 5. ויזואליזציה --------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES):
    cm = build_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.set_title('Confusion Matrix (4x4) - Iris Classifier', fontsize=14, pad=12)
    plt.tight_layout()
    path = os.path.join(DOCS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved -> {path}")


def plot_convergence(ce_history, mse_history):
    """גרף התכנסות כפול: Cross-Entropy (שמאל) ו-MSE (ימין)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(ce_history) + 1)

    # -- Cross-Entropy --
    ax1.plot(epochs_range, ce_history, color='royalblue', linewidth=1.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Convergence - Cross-Entropy Loss', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.annotate(f'Final: {ce_history[-1]:.4f}',
                 xy=(len(ce_history), ce_history[-1]),
                 xytext=(len(ce_history) * 0.55, ce_history[0] * 0.6),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    # -- MSE --
    ax2.plot(epochs_range, mse_history, color='darkorange', linewidth=1.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE Loss', fontsize=12)
    ax2.set_title('Convergence - MSE  [E_MSE = 1/N * sum((y - y_tilde)^2)]', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.annotate(f'Final: {mse_history[-1]:.5f}',
                 xy=(len(mse_history), mse_history[-1]),
                 xytext=(len(mse_history) * 0.55, mse_history[0] * 0.6),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    plt.suptitle('Convergence Graph - Iris Classifier MLP', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(DOCS_DIR, 'convergence_graph.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved -> {path}")


# --- 6. Main ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("   Iris_Classifier_Pro v2 - Training & Evaluation")
    print("   [1] StandardScaler  [2] 2D-Split  [3] MLP+ReLU  [4] CE")
    print("=" * 60)

    # 1. טעינת נתונים
    X, y = load_and_prepare_data()
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n[Data] Classes      : {CLASS_NAMES}")
    print(f"[Data] Distribution : { {CLASS_NAMES[k]: int(v) for k, v in zip(unique, counts)} }")

    # 2. חלוקת Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"[Data] Train: {len(X_train)} samples | Test: {len(X_test)} samples (80%/20%)")

    # 3. שמירת CSV
    save_datasets(X, y, X_train, X_test, y_train, y_test)

    # 4. נרמול (שיפור 1)
    X_train_s, X_test_s, _, _ = standard_scale(X_train, X_test)
    print("[Prep] Features standardized (z-score) - Improvement #1")

    # 5. One-Hot
    Y_train_oh = one_hot_encode(y_train)

    # 6. אימון (שיפורים 3+4)
    print(f"\n[Train] MLP: 4 -> {HIDDEN_SIZE}(ReLU) -> 4(Softmax)"
          f" | lr={LEARNING_RATE} | epochs={EPOCHS}")
    print("-" * 60)
    weights, ce_history, mse_history = train(X_train_s, Y_train_oh)

    # 7. הערכה
    y_pred   = predict(X_test_s, weights)
    accuracy = compute_accuracy(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  [Result] Test Accuracy : {accuracy * 100:.2f}%")
    print(f"  [Result] Final CE Loss : {ce_history[-1]:.6f}")
    print(f"  [Result] Final MSE     : {mse_history[-1]:.6f}")
    print("=" * 60)

    # 8. גרפים
    os.makedirs(DOCS_DIR, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred)
    plot_convergence(ce_history, mse_history)

    print("\n[Done] All outputs saved to docs/")
    print("=" * 60)


if __name__ == '__main__':
    main()
