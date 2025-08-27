.PHONY: all toys adult covtype cm ablations clean

PY=python

all: toys adult covtype cm ablations

toys:
	$(PY) experiments/run_sklearn_toys.py

adult:
	$(PY) experiments/run_adult_autogluon.py
	$(PY) experiments/run_adult_sklearn.py

covtype:
	$(PY) experiments/run_covtype_autogluon.py
	$(PY) experiments/run_covtype_sklearn.py

cm:
	$(PY) experiments/make_confusion_matrices.py

ablations:
	$(PY) experiments/run_autogluon_ablations.py

clean:
	rm -rf AutogluonModels
