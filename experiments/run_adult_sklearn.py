
import os, time, numpy as np, pandas as pd
from openml import datasets as oml_datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import append_csv, RESULTS_DIR

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

adult = oml_datasets.get_dataset(1590)  # adult
Xy, _, _, _ = adult.get_data(dataset_format="dataframe", target=adult.default_target_attribute)
df = Xy.dropna().reset_index(drop=True).rename(columns={adult.default_target_attribute: "target"})

X = df.drop(columns=["target"]); y = df["target"]
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.columns.difference(cat_cols).tolist()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

def run_model(name, estimator, grid, preproc):
    pipe = Pipeline([("prep", preproc), ("clf", estimator)])
    t0 = time.time()
    hs = HalvingGridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True, factor=3, verbose=0)
    hs.fit(X_tr, y_tr)
    t1 = time.time()
    acc = accuracy_score(y_te, hs.best_estimator_.predict(X_te))
    append_csv(RESULTS_DIR / "adult_accuracy.csv", {"model": name, "accuracy": f"{acc:.4f}"})
    append_csv(RESULTS_DIR / "adult_time.csv", {"model": name, "time_sec": f"{t1 - t0:.2f}"})
    print(f"Adult — {name}: accuracy={acc:.4f}, time={t1 - t0:.2f}s")

preproc_dense = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                                   ("num", StandardScaler(), num_cols)], remainder="drop")
preproc_sparse = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                    ("num", "passthrough", num_cols)], remainder="drop", sparse_threshold=1.0)

run_model("RF (HalvingGrid)", RandomForestClassifier(random_state=RANDOM_STATE),
          {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 20], "clf__min_samples_leaf": [1, 2]},
          preproc_sparse)

run_model("ET (HalvingGrid)", ExtraTreesClassifier(random_state=RANDOM_STATE),
          {"clf__n_estimators": [300, 600], "clf__max_depth": [None, 20]},
          preproc_sparse)

def fit_direct(name, pipe):
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    t1 = time.time()
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_te, pipe.predict(X_te))
    append_csv(RESULTS_DIR / "adult_accuracy.csv", {"model": name, "accuracy": f"{acc:.4f}"})
    append_csv(RESULTS_DIR / "adult_time.csv", {"model": name, "time_sec": f"{t1 - t0:.2f}"})
    print(f"Adult — {name}: accuracy={acc:.4f}, time={t1 - t0:.2f}s")

voters = VotingClassifier(estimators=[
    ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("et", ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE)),
    ("lr", LogisticRegression(max_iter=1000, solver="liblinear"))
], voting="soft")

stack = StackingClassifier(estimators=[
    ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("et", ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE)),
    ("lr", LogisticRegression(max_iter=1000, solver="liblinear"))
], final_estimator=LogisticRegression(max_iter=1000, solver="liblinear"))

fit_direct("VotingSoft (direct)", Pipeline([("prep", preproc_dense), ("clf", voters)]))
fit_direct("Stacking (direct)", Pipeline([("prep", preproc_dense), ("clf", stack)]))

run_model("LogReg (HalvingGrid)", LogisticRegression(max_iter=1000, solver="liblinear"),
          {"clf__C": [0.1, 1.0, 10.0]}, preproc_dense)
