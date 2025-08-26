
import os, time, pandas as pd
from openml import datasets as oml_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor
from utils import append_csv, RESULTS_DIR

TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "300"))
RANDOM_STATE = 42

adult = oml_datasets.get_dataset(1590)  # adult
Xy, _, _, _ = adult.get_data(dataset_format="dataframe", target=adult.default_target_attribute)
df = Xy.dropna().reset_index(drop=True).rename(columns={adult.default_target_attribute: "target"})

X = df.drop(columns=["target"]); y = df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

train_df = X_tr.copy(); train_df["target"] = y_tr
test_df  = X_te.copy(); test_df["target"]  = y_te

t0 = time.time()
predictor = TabularPredictor(label="target", eval_metric="accuracy") \
    .fit(train_df, time_limit=TIME_LIMIT, presets="best_quality", verbosity=0)
t1 = time.time()

acc = accuracy_score(y_te, predictor.predict(X_te))
append_csv(RESULTS_DIR / "adult_accuracy.csv", {"model": "AutoGluon", "accuracy": f"{acc:.4f}"})
append_csv(RESULTS_DIR / "adult_time.csv", {"model": "AutoGluon", "time_sec": f"{t1 - t0:.2f}"})
print(f"Adult — AutoGluon: accuracy={acc:.4f}, time={t1 - t0:.2f}s")
