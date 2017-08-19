import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split

import pickle

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

# LAT_NORM = 40.7
# LON_NORM = -73.9
#
# df_train['pickup_latitude'] -= LAT_NORM
# df_train['dropoff_latitude'] -= LAT_NORM
# df_train['pickup_longitude'] -= LON_NORM
# df_train['dropoff_longitude'] -= LON_NORM
# df_test['pickup_latitude'] -= LAT_NORM
# df_test['dropoff_latitude'] -= LAT_NORM
# df_test['pickup_longitude'] -= LON_NORM
# df_test['dropoff_longitude'] -= LON_NORM

print('Loaded data ...')

id_test = df_test['id']
y_train = np.log1p(df_train['trip_duration'])

keep_ftrs = ['vendor_id', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', ]
x_train = df_train[keep_ftrs]
x_test = df_test[keep_ftrs]

for dataset in [df_train, df_test]:
    dataset['day'] = dataset['pickup_datetime'].apply(lambda x: x.split(' ')[0])
    dataset['hour'] = dataset['pickup_datetime'].apply(lambda x: x.split(':')[0])

df_all = pd.concat([df_train, df_test])
daily_counts = df_all.groupby('day')['day'].count()
hourly_counts = df_all.groupby('hour')['hour'].count()
print(hourly_counts)

df_train['manhattan_speed'] = (np.abs(df_train['pickup_latitude'] - df_train['dropoff_latitude']) + np.abs(df_train['pickup_longitude'] - df_train['dropoff_longitude'])) / (df_train['trip_duration'])
hourly_speed = df_train.groupby('hour')['manhattan_speed'].mean()
pickle.dump(hourly_speed, open('minidump.bin', 'wb'))
manhattan_speed_fill = df_train['manhattan_speed'].mean()
print(hourly_speed)

for dataset, orig in [(x_train, df_train), (x_test, df_test)]:
    dataset['delta_lat'] = dataset['pickup_latitude'] - dataset['dropoff_latitude']
    dataset['delta_lon'] = dataset['pickup_longitude'] - dataset['dropoff_longitude']
    dataset['manhattan'] = np.abs(dataset['delta_lat']) + np.abs(dataset['delta_lon'])
    dataset['euclidean'] = np.sqrt(dataset['delta_lat'] ** 2 + dataset['delta_lon'] ** 2)
    dataset['daily_count'] = orig['pickup_datetime'].apply(lambda x: daily_counts[x.split(' ')[0]])
    dataset['hourly_count'] = orig['pickup_datetime'].apply(lambda x: hourly_counts[x.split(':')[0]])
    dataset['hourly_speed'] = orig['hour'].apply(lambda x: hourly_speed[x] if x in hourly_speed else manhattan_speed_fill)
    dataset['hourly_speed_estim'] = dataset['hourly_speed'] * dataset['manhattan']

    dataset['direction'] = dataset[['delta_lat', 'delta_lon']].apply(lambda x: np.arctan2(x[0], x[1]))

    print('processing time ...')
    time = pd.to_datetime(orig['pickup_datetime'])
    dataset['day_of_week'] = [t.dayofweek for t in time]
    dataset['epoch_time'] = [np.int64(t.value) / 1000000000 for t in time]
    dataset['hour'] = [t.hour for t in time]

print(x_train)

print(np.corrcoef(df_train['trip_duration'].values, x_train['hourly_speed_estim'].values))

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=4242)

print('Creating DMatrices...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

params = {}
params['eta'] = 0.05
params['objective'] = 'reg:linear'
params['eval_metric'] = 'rmse'
params['colsample_bylevel'] = 0.8
params['subsample'] = 0.8
params['max_depth'] = 12
params['min_child_weight'] = 2
params['silent'] = 1

print('Training ...')

reg = xgb.train(params, d_train, 5000, watchlist, early_stopping_rounds=100)

feature_importance_dict = reg.get_fscore()
pickle.dump(feature_importance_dict, open('dump.bin', 'wb'))

p_test = np.expm1(reg.predict(xgb.DMatrix(x_test)))

sub = pd.DataFrame()
sub['id'] = id_test
sub['trip_duration'] = p_test
sub.to_csv('test_sub.csv', index=False)
