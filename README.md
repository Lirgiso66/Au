# AutoML Thesis Experiments
- **Release (заморожённая версия результатов):** https://github.com/Lirgiso66/Au/releases/tag/v1.0
[![Open In Colab](https://colab.research.googleusercontent.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lirgiso66/Au/blob/main/colab_quickstart.ipynb)

Reproduce:
```bash
python -m pip install -U pip
pip install -r requirements.txt
make all   # or run scripts one by one (see below)


This repository contains scripts to reproduce the experiments for the thesis:
- **Datasets:** Wine, Breast Cancer, Digits, Iris, Adult Income, Covertype
- **Systems:** AutoGluon (Tabular) vs classic scikit-learn pipelines and ensembles
- **Extras:** Confusion matrices, ablations for AutoGluon (stacking/time budget)

## How to run (quick)
```bash
python -m pip install -U pip
pip install -r requirements.txt
# Datasets from sklearn/openml are downloaded automatically

# 1) Classic sklearn baselines + ensembles on 4 toy datasets
python experiments/run_sklearn_toys.py

# 2) Adult Income: AutoGluon + sklearn baselines
python experiments/run_adult_autogluon.py
python experiments/run_adult_sklearn.py

# 3) Covertype: AutoGluon + sklearn baselines
python experiments/run_covtype_autogluon.py
python experiments/run_covtype_sklearn.py

# 4) Confusion matrices for the 4 toy datasets (best sklearn vs AG)
python experiments/make_confusion_matrices.py

# 5) AutoGluon ablations (no stacking / half time)
python experiments/run_autogluon_ablations.py
```

All results are saved into `results/` as CSVs and PNGs ready to be included into LaTeX.

## Notes
- To make sklearn search "time-aware", we use **small grids** and **Successive Halving** (`HalvingGridSearchCV`) where reasonable, as a practical budget proxy.
- Set `TIME_LIMIT` env var to control AG budget (seconds). Defaults in scripts are 300s for Adult/Covtype.
- For Covertype on CPU, consider subsampling to 50k rows (see commented lines).
- И маленькая сводная табличка (под «Notes»), чтобы научруку было понятно глазами:
```markdown
### Quick results (from `results/*.csv`)
| Dataset | Best accuracy | System | Train time (s) |
|--------:|---------------|--------|----------------|
| Adult   | **0.8633**    | AutoGluon | 35.6 |
| Covtype (subsample) | **0.8983** | AutoGluon | 82.9 |

See `results/` for full CSVs and `results/figures/` for confusion matrices.

