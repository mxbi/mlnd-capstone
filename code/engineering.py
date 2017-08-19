import numpy as np
import pandas as pd
import pickle
import multiprocessing

import functions

# The number of CPU threads to use for multithreaded operations
N_THREADS = 8

# Distance function definitions
def manhattan_distance(a_lat, a_lon, b_lat, b_lon):
    return np.abs(a_lon - b_lon) + np.abs(a_lat - b_lat)

def euclidean_distance(a_lat, a_lon, b_lat, b_lon):
    return np.sqrt((a_lon - b_lon) ** 2 + (a_lat - b_lat) ** 2)

def haversine_distance(x):
    a_lat, a_lon, b_lat, b_lon = x
    return haversine.haversine((a_lat, a_lon), (b_lat, b_lon))

# Multithreaded apply function for a dataframe. This uses multiprocessing to map a function to a series, vastly speeding up feature generation
def apply_multithreaded(data, func):
    pool = multiprocessing.Pool(N_THREADS)  # Spawn a pool of processes
    data = data.values  # Retrieve a numpy array which can be iterated over

    result = pool.map(func, data)  # Map the function over the data multi-threaded
    pool.close()  # Close the threads
    return result

# This code needs to be removed from global scope, else it will be run by every thread in the mulitprocessing pool when they import this file.
if __name__ == '__main__':
    print('Loading preprocessed data ...')

    df_train, df_test, y_train, id_test = pickle.load(open('preprocessed_data.bin', 'rb'))

    print(df_train.columns)

    # First, we construct x_train and x_test dataframes. These contain ONLY the features we use for training
    drop_features = ['pickup_datetime']
    x_train = df_train.drop(drop_features, axis=1)
    x_test = df_test.drop(drop_features, axis=1)

    print('Creating distance features ...')

    # Map the three distance functions over all samples in the training set
    x_train['dist_l1'] = np.abs(x_train['pickup_latitude'] - x_train['dropoff_latitude']) + np.abs(x_train['pickup_longitude'] - x_train['dropoff_longitude'])
    x_train['dist_l2'] = np.sqrt((x_train['pickup_latitude'] - x_train['dropoff_latitude']) ** 2 + (x_train['pickup_longitude'] - x_train['dropoff_longitude']) ** 2)
    x_train['dist_haversine'] = apply_multithreaded(x_train[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']], functions.haversine_distance)

    x_test['dist_l1'] = np.abs(x_test['pickup_latitude'] - x_test['dropoff_latitude']) + np.abs(x_test['pickup_longitude'] - x_test['dropoff_longitude'])
    x_test['dist_l2'] = np.sqrt((x_test['pickup_latitude'] - x_test['dropoff_latitude']) ** 2 + (x_test['pickup_longitude'] - x_test['dropoff_longitude']) ** 2)
    x_test['dist_haversine'] = apply_multithreaded(x_test[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']], functions.haversine_distance)

    print(x_train[['dist_l1', 'dist_l2', 'dist_haversine']].head())

    print('Creating direction of travel features ...')
    x_train['delta_lat'] = df_train['dropoff_latitude'] - df_train['pickup_latitude']
    x_train['delta_lon'] = df_train['dropoff_longitude'] - df_train['pickup_longitude']
