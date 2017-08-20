import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
import xgboost as xgb

def main():
    print('Loading data from disk ...')
    x_train, x_test, y_train, id_test = pickle.load(open('engineered_data.bin', 'rb'))

    print('Loaded {} features.'.format(x_train.shape[1]))

    # First we take the log1p of the target value
    y_train = np.log1p(y_train)

    # We split off a validation set
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=1337)

    print('{} training samples, {} validation samples'.format(len(x_train), len(x_valid)))
    print('Constructing XGBoost DMatrices ...')

    # Convert our data into XGBoost's fast C++ format
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    d_test = xgb.DMatrix(x_test)

    # Remove these from memory as they are unncessary now
    del x_train, x_valid, x_test

    # When computing metrics, XGBoost will do it on these two datasets
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Setup parameters for XGBoost training
    params = {}
    params['objective'] = 'reg:linear'  # This is a regression problem
    params['eval_metric'] = 'rmse'      # We will use RMSE for evaluation - after the log1p operation we did this becomes RMSLE
    params['silent'] = 1                # Don't print debug messages
    params['eta'] = 0.1                # Learning reduced to a more reasonable rate - this does not need to be tuned as lower is always better
    # All other parameters left at defaults

    print('Training XGBoost model ...')

    # Early stop after 50 boosting rounds
    reg = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=10)

    # Predict on the test set
    p_test = reg.predict(d_test)

    # Create a submission file
    sub = pd.DataFrame()
    sub['id'] = id_test
    sub['trip_duration'] = np.expm1(p_test)

    # Write the csv to disk
    sub.to_csv('model_sub.csv', index=False)

if __name__ == '__main__':
    main()
