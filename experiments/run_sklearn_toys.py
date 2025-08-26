
import time, numpy as np, pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import append_csv, RESULTS_DIR

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
out_csv_acc = RESULTS_DIR / "toys_sklearn_accuracy.csv"
out_csv_time = RESULTS_DIR / "toys_sklearn_time.csv"

def dataset_loader(name):
    if name == "wine":
        data = datasets.load_wine()
    elif name == "breast":
        data = datasets.load_breast_cancer()
    elif name == "digits":
        data = datasets.load_digits()
    elif name == "iris":
        data = datasets.load_iris()
    else:
        raise ValueError("unknown dataset")
    X = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, "feature_names") else None)
    y = pd.Series(data.target, name="target")
    return X, y

def run_one(name):
    X, y = dataset_loader(name)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    num_cols = list(X_tr.columns)
    scaler = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")

    models = {
        "RF": Pipeline([("prep", "passthrough"), ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]),
        "ET": Pipeline([("prep", "passthrough"), ("clf", ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE))]),
        "GB": Pipeline([("prep", "passthrough"), ("clf", GradientBoostingClassifier())]),
        "HGB": Pipeline([("prep", "passthrough"), ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
        "LR": Pipeline([("prep", scaler), ("clf", LogisticRegression(max_iter=1000))]),
        "KNN": Pipeline([("prep", scaler), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }

    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("hgb", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ("lr", LogisticRegression(max_iter=1000)),
    ]
    models["Stacking"] = Pipeline([("prep", scaler),
                                   ("clf", StackingClassifier(estimators=base_estimators,
                                                              final_estimator=LogisticRegression(max_iter=1000),
                                                              passthrough=False))])
    models["VotingSoft"] = Pipeline([("prep", scaler),
                                     ("clf", VotingClassifier(estimators=[
                                         ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
                                         ("hgb", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
                                         ("lr", LogisticRegression(max_iter=1000)),
                                     ], voting="soft"))])

    for mname, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        t1 = time.time()
        acc = accuracy_score(y_te, pipe.predict(X_te))
        append_csv(out_csv_acc, {"dataset": name, "model": mname, "accuracy": f"{acc:.4f}"})
        append_csv(out_csv_time, {"dataset": name, "model": mname, "time_sec": f"{t1 - t0:.2f}"})
        print(f"{name} — {mname}: acc={acc:.4f}, time={t1 - t0:.2f}s")

if __name__ == "__main__":
    for ds in ["wine", "breast", "digits", "iris"]:
        run_one(ds)
    print("Saved:", out_csv_acc, "and", out_csv_time)
