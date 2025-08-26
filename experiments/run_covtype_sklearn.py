
import time, numpy as np, pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import append_csv, RESULTS_DIR

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

cov = fetch_covtype(as_frame=True)
X_full = cov.frame.drop(columns=['Cover_Type'])
y_full = pd.Series(cov.target, name='target')

X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE)

num_cols = X_tr.columns.tolist()

def run_model(name, estimator, grid, preproc):
    pipe = Pipeline([("prep", preproc), ("clf", estimator)])
    t0 = time.time()
    hs = HalvingGridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True, factor=3, verbose=0)
    hs.fit(X_tr, y_tr)
    t1 = time.time()
    acc = accuracy_score(y_te, hs.best_estimator_.predict(X_te))
    append_csv(RESULTS_DIR / "covtype_accuracy.csv", {"model": name, "accuracy": f"{acc:.4f}"})
    append_csv(RESULTS_DIR / "covtype_time.csv", {"model": name, "time_sec": f"{t1 - t0:.2f}"})
    print(f"Covertype — {name}: accuracy={acc:.4f}, time={t1 - t0:.2f}s")

preproc = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")

run_model("RF (HalvingGrid)", RandomForestClassifier(random_state=RANDOM_STATE),
          {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 30], "clf__min_samples_leaf": [1,2]},
          "passthrough")

run_model("ET (HalvingGrid)", ExtraTreesClassifier(random_state=RANDOM_STATE),
          {"clf__n_estimators": [400, 800], "clf__max_depth": [None, 30]},
          "passthrough")

run_model("HGB (HalvingGrid)", HistGradientBoostingClassifier(random_state=RANDOM_STATE),
          {"clf__max_depth": [None, 10], "clf__learning_rate": [0.05, 0.1]},
          preproc)

stack = StackingClassifier(estimators=[
    ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("et", ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE)),
    ("hgb", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
], final_estimator=LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs"))

vote = VotingClassifier(estimators=[
    ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("et", ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE)),
    ("hgb", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
], voting="soft")

def fit_direct(name, pipe):
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    t1 = time.time()
    acc = accuracy_score(y_te, pipe.predict(X_te))
    append_csv(RESULTS_DIR / "covtype_accuracy.csv", {"model": name, "accuracy": f"{acc:.4f}"})
    append_csv(RESULTS_DIR / "covtype_time.csv", {"model": name, "time_sec": f"{t1 - t0:.2f}"})
    print(f"Covertype — {name}: accuracy={acc:.4f}, time={t1 - t0:.2f}s")

fit_direct("Stacking (direct)", Pipeline([("prep", preproc), ("clf", stack)]))
fit_direct("VotingSoft (direct)", Pipeline([("prep", preproc), ("clf", vote)]))
