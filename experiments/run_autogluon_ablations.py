
import os, time, pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor
from utils import append_csv, RESULTS_DIR

TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "180"))
RANDOM_STATE = 42
out_csv = RESULTS_DIR / "ablations_autogluon.csv"

def run_on(name, X, y, no_stack=False, half_time=False):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    train_df = pd.DataFrame(X_tr).copy(); train_df['target'] = y_tr
    test_df  = pd.DataFrame(X_te).copy(); test_df['target']  = y_te

    params = dict(time_limit=TIME_LIMIT, presets="best_quality", verbosity=0)
    tag = "default"
    if no_stack:
        params.update(num_stack_levels=0)  # disable stacking
        tag = "no_stack"
    if half_time:
        params["time_limit"] = max(60, TIME_LIMIT // 2)
        tag = f"{tag}_half_time"

    t0 = time.time()
    predictor = TabularPredictor(label='target', eval_metric='accuracy').fit(train_df, **params)
    t1 = time.time()
    acc = accuracy_score(y_te, predictor.predict(pd.DataFrame(X_te)))
    append_csv(out_csv, {"dataset": name, "variant": tag, "accuracy": f"{acc:.4f}", "time_sec": f"{t1 - t0:.2f}"})

if __name__ == "__main__":
    for name, loader in [("wine", datasets.load_wine), ("digits", datasets.load_digits)]:
        d = loader()
        run_on(name, d.data, d.target, no_stack=False, half_time=False)
        run_on(name, d.data, d.target, no_stack=True,  half_time=False)
        run_on(name, d.data, d.target, no_stack=False, half_time=True)
    print("Saved:", out_csv)
