
import os, time, pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor
from utils import append_csv, RESULTS_DIR

TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "300"))
RANDOM_STATE = 42

cov = fetch_covtype(as_frame=True)
X_full = cov.frame.drop(columns=['Cover_Type'])
y_full = pd.Series(cov.target, name='target').astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE)
train_df = X_tr.copy(); train_df['target'] = y_tr
test_df  = X_te.copy(); test_df['target']  = y_te

t0 = time.time()
predictor = TabularPredictor(label='target', eval_metric='accuracy') \
    .fit(train_df, time_limit=TIME_LIMIT, presets='best_quality', verbosity=0)
t1 = time.time()

y_pred = predictor.predict(X_te)
acc = accuracy_score(y_te.astype(str), pd.Series(y_pred).astype(str))

append_csv(RESULTS_DIR / "covtype_accuracy.csv", {"model": "AutoGluon", "accuracy": f"{acc:.4f}"})
append_csv(RESULTS_DIR / "covtype_time.csv", {"model": "AutoGluon", "time_sec": f"{t1 - t0:.2f}"})
print(f"Covertype — AutoGluon: accuracy={acc:.4f}, time={t1 - t0:.2f}s")
