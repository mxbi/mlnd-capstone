# mlnd-capstone

### Requirements
`numpy` (included in Anaconda)
`pandas` (included in Anaconda)
`scikit-learn` (included in Anaconda)
`haversine` (can be installed with `pip install haversine`)
`xgboost==0.6` (can be installed with `pip install xgboost` on Linux - on windows it can be obtained from Anaconda cloud)

### Dataset
The dataset can be obtained from the Kaggle competition page: https://www.kaggle.com/c/nyc-taxi-trip-duration/data (signup required)
`train.csv` and `test.csv` must be unzipped and placed in the `data/` folder in this repository for code to run.

### Code running order
From inside the code/ folder:
1. `python processing.py`
2. `python engineering.py`
3. `python model.py` (my initial model)
4. `python model_search.py` (parameter optimisation)
5. `python model_optim.py` (the final model)
