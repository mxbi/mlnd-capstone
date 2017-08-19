import pandas as pd
import numpy as np
import pickle

print('Reading data into memory ...')

# Read in the input dataset
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

print('Read {} training data and {} testing data, preprocessing ...'.format(df_train.shape, df_test.shape))

# Convert store_and_fwd_flag into a binary numeric feature
df_train.loc[:, 'store_and_fwd_flag'] = df_train['store_and_fwd_flag'].apply(lambda x: int(x == 'Y'))
df_test.loc[:, 'store_and_fwd_flag'] = df_test['store_and_fwd_flag'].apply(lambda x: int(x == 'Y'))

# Parse pickup_datetime feature
train_datetime = pd.to_datetime(df_train['pickup_datetime'])
test_datetime = pd.to_datetime(df_test['pickup_datetime'])

# Create unix epoch time feature
# It is usually in nanoseconds, so we convert to seconds for readability
df_train['unix_time'] = [np.int64(t.value) / 1000000000 for t in train_datetime]
df_test['unix_time'] = [np.int64(t.value) / 1000000000 for t in test_datetime]

# Create minute of day feature
df_train['daily_minute'] = [t.hour*60 + t.minute for t in train_datetime]
df_test['daily_minute'] = [t.hour*60 + t.minute for t in test_datetime]

# Create day of week feature
df_train['day_of_week'] = [t.dayofweek for t in train_datetime]
df_test['day_of_week'] = [t.dayofweek for t in test_datetime]

# Extract target variable and keep it in memory
# Also do the same with test ID, because they are required for the Kaggle submission
y_train = df_train['trip_duration'].values
id_test = df_test['id'].values

# Drop 'dropoff_datetime' feature from training set, as well as IDs and y values
df_train.drop(['dropoff_datetime', 'id', 'trip_duration'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

print(df_train.head())
print('Preprocessing complete, saving data to disk ...')

# Use pickle to dump our working data to disk so we can use it again in the future.
pickle.dump([df_train, df_test, y_train, id_test], open('preprocessed_data.bin', 'wb'), protocol=2)
