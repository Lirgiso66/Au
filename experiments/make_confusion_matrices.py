
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def plot_cm(cm, title, path):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def one_dataset(name):
    if name == "wine":
        data = datasets.load_wine()
    elif name == "breast":
        data = datasets.load_breast_cancer()
    elif name == "digits":
        data = datasets.load_digits()
    elif name == "iris":
        data = datasets.load_iris()
    else:
        raise ValueError

    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_tr, y_tr)
    cm = confusion_matrix(y_te, clf.predict(X_te))
    out = OUT / f"cm_{name}_sklearn.png"
    plot_cm(cm, f"{name} — sklearn RF", out)
    print("Saved", out)

if __name__ == "__main__":
    for ds in ["wine", "breast", "digits", "iris"]:
        one_dataset(ds)
