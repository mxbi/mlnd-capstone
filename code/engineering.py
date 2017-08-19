import numpy as np
import pandas as pd
import pickle
import multiprocessing
import haversine

import functions

# The number of CPU threads to use for multithreaded operations
N_THREADS = 8

# Haversine distance function definition
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

    df_train, df_test, x_train, x_test, y_train, id_test = pickle.load(open('preprocessed_data.bin', 'rb'))

    print(df_train.columns)

    print('Creating distance features ...')

    # Map the three distance functions over all samples in the training set
    x_train['dist_l1'] = np.abs(x_train['pickup_latitude'] - x_train['dropoff_latitude']) + np.abs(x_train['pickup_longitude'] - x_train['dropoff_longitude'])
    x_train['dist_l2'] = np.sqrt((x_train['pickup_latitude'] - x_train['dropoff_latitude']) ** 2 + (x_train['pickup_longitude'] - x_train['dropoff_longitude']) ** 2)
    # As haversine is not vectorised, we use the multtithreading approach for speed
    x_train['dist_haversine'] = apply_multithreaded(x_train[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']], functions.haversine_distance)

    x_test['dist_l1'] = np.abs(x_test['pickup_latitude'] - x_test['dropoff_latitude']) + np.abs(x_test['pickup_longitude'] - x_test['dropoff_longitude'])
    x_test['dist_l2'] = np.sqrt((x_test['pickup_latitude'] - x_test['dropoff_latitude']) ** 2 + (x_test['pickup_longitude'] - x_test['dropoff_longitude']) ** 2)
    x_test['dist_haversine'] = apply_multithreaded(x_test[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']], functions.haversine_distance)

    print(x_train[['dist_l1', 'dist_l2', 'dist_haversine']].head())

    print('Creating direction of travel features ...')

    x_train['delta_lat'] = x_train['dropoff_latitude'] - x_train['pickup_latitude']
    x_train['delta_lon'] = x_train['dropoff_longitude'] - x_train['pickup_longitude']
    x_train['angle'] = (180 / np.pi) * np.arctan2(x_train['delta_lat'], x_train['delta_lon']) + 180

    x_test['delta_lat'] = x_test['dropoff_latitude'] - x_test['pickup_latitude']
    x_test['delta_lon'] = x_test['dropoff_longitude'] - x_test['pickup_longitude']
    x_test['angle'] = (180 / np.pi) * np.arctan2(x_test['delta_lat'], x_test['delta_lon']) + 180

    print(x_train[['delta_lat', 'delta_lon', 'angle']].head())

    print('Creating traffic features ...')

    # First, we extract the day and hour from each datetime string
    df_train['day'] = df_train['pickup_datetime'].apply(lambda x: x.split(' ')[0])
    df_train['hour'] = df_train['pickup_datetime'].apply(lambda x: x.split(':')[0])
    df_test['day'] = df_test['pickup_datetime'].apply(lambda x: x.split(' ')[0])
    df_test['hour'] = df_test['pickup_datetime'].apply(lambda x: x.split(':')[0])

    # Apply a groupby operation over unique dates in order to get the number of trips on each day
    df_all = pd.concat([df_train[['day', 'hour']], df_test[['day', 'hour']]])  # Combine the two datasets so metrics can be computed over all the data
    daily_traffic = df_all.groupby('day')['day'].count()  # Count the number of trips on each day
    hourly_traffic = df_all.groupby('hour')['hour'].count()  # Count the number of trips in each hour

    print(daily_traffic.head())
    print(hourly_traffic.head())

    # Loop over the data and lookup the count information on the corresponding day to fill the feature
    x_train['daily_count'] = df_train['day'].apply(lambda day: daily_traffic[day])
    x_train['hourly_count'] = df_train['hour'].apply(lambda hour: hourly_traffic[hour])
    x_test['daily_count'] = df_test['day'].apply(lambda day: daily_traffic[day])
    x_test['hourly_count'] = df_test['hour'].apply(lambda hour: hourly_traffic[hour])

    print(x_train[['daily_count', 'hourly_count']].head())

    print('Creating time estimate features ...')

    df_train['haversine_speed'] = x_train['dist_haversine'] / df_train['trip_duration']  # Calculate haversine speed for training set
    hourly_speed = df_train.groupby('hour')['haversine_speed'].mean()  # Find average haversine_speed for each hour in the training set
    hourly_speed_fill = df_train['haversine_speed'].mean()  # Get mean across whole dataset for filling unknowns

    # Create feature
    train_hourly_speed = df_train['hour'].apply(lambda hour: hourly_speed[hour])
    test_hourly_speed = df_test['hour'].apply(lambda hour: hourly_speed[hour] if hour in hourly_speed else hourly_speed_fill)
    x_train['haversine_speed_estim'] = x_train['dist_haversine'] / train_hourly_speed
    x_test['haversine_speed_estim'] = x_test['dist_haversine'] / test_hourly_speed

    print(x_train['haversine_speed_estim'].head())

    print('Feature engineering complete, feature list: {}'.format(x_train.columns.tolist()))
    print('Serialising data to disk ...')

    pickle.dump([x_train, x_test, y_train, id_test], open('engineered_data.bin', 'wb'), protocol=2)
