import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
import xgboost as xgb

def run_model(d_train, d_valid, params):
    # When computing metrics, XGBoost will do it on these two datasets
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # This dictionary stores training results
    res_dict = {}

    # Early stop after 50 boosting rounds
    reg = xgb.train(params, d_train, 5, watchlist, early_stopping_rounds=10, evals_result=res_dict)

    return np.min(res_dict['valid']['rmse'])

def optimise_parameter(d_train, d_valid, name, space, params):
    print('Optimising {}'.format(name))

    scores = []
    for trial in space:
        params[name] = trial
        score = run_model(d_train, d_valid, params)
        print('Finished running with {} value {}, score {}'.format(name, trial, score))
        scores.append(score)

    print(zip(space, scores))
    return scores, np.array(space)[np.argmax(scores)]  # Return best value for parameter

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

    # Setup parameters for XGBoost training
    params = {}
    params['objective'] = 'reg:linear'  # This is a regression problem
    params['eval_metric'] = 'rmse'      # We will use RMSE for evaluation - after the log1p operation we did this becomes RMSLE
    params['silent'] = 1                # Don't print debug messages
    params['eta'] = 0.1                # Learning reduced to a more reasonable rate - this does not need to be tuned as lower is always

    # I begin with a educated guess of what will work well
    params['max_depth'] = 6
    params['colsample_bylevel'] = 0.9
    params['subsample'] = 0.9
    params['min_child_weight'] = 1

    # Search space
    max_depth = [4, 5, 6, 8, 10, 12, 14]
    colsample_bylevel = [1, 0.9, 0.8, 0.7]
    subsample = [1, 0.9, 0.8, 7]
    min_child_weight = [0.5, 1, 2, 3]

    # Find the best  max_depth, saving results for each max_depth value
    max_depth_scores, params['max_depth'] = optimise_parameter(d_train, d_valid, 'max_depth', max_depth, params)
    colsample_bylevel_scores, params['colsample_bylevel'] = optimise_parameter(d_train, d_valid, 'colsample_bylevel', colsample_bylevel, params)
    subsample_scores, params['subsample'] = optimise_parameter(d_train, d_valid, 'subsample', subsample, params)
    min_child_weight_scores, params['min_child_weight'] = optimise_parameter(d_train, d_valid, 'min_child_weight', min_child_weight, params)

    print(max_depth_Scores, colsample_bylevel_scores, subsample_scores, min_child_weight_scores)

if __name__ == '__main__':
    main()
