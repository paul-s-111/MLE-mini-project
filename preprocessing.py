from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import copy
import numpy as np
import pandas as pd

def preprocess_one_hot(data):
    # Extract features from the "datetime" column
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['Hour'] = data.index.hour

    # Create a month category column for different days
    data['month_cat'] = np.select(
        [(data['Day'] <= 10), (data['Day'] > 10) & (data['Day'] <= 20), (data['Day'] > 20)],
        ['beginning', 'middle', 'end'],
        default='unknown_category'
    )

    # Perform one-hot encoding for month and month categories (days)
    data_encoded = pd.get_dummies(data, columns=['Month', 'month_cat'])

    # Specify specific labels for Hour and set "nighttime" when no power is ever generated
    hour_labels = {
        1: '01', 2: '02', 3: '03', 4: '04', 5: '05',
        6: '06', 7: '07', 8: '08', 9: '09', 10: 'nighttime',
        11: 'nighttime', 12: 'nighttime', 13: 'nighttime',
        14: 'nighttime', 15: 'nighttime', 16: 'nighttime',
        17: 'nighttime', 18: 'nighttime', 19: 'nighttime',
        20: '20', 21: '21', 22: '22', 23: '23', 0: '00'
    }

    data_encoded['Hour'] = data_encoded['Hour'].map(hour_labels)

    # Perform one-hot encoding for Hour
    data_encoded = pd.get_dummies(data_encoded, columns=['Hour'])

    return data_encoded

def preprocess_convert(data):
    # Convert the object columns to boolean
    columns_to_convert = ['Month_1', 'Month_2', 'Month_3', 'Month_4',
       'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
       'Month_11', 'Month_12', 'month_cat_beginning', 'month_cat_end',
       'month_cat_middle', 'Hour_00', 'Hour_01', 'Hour_02', 'Hour_03',
       'Hour_04', 'Hour_05', 'Hour_06', 'Hour_07', 'Hour_08', 'Hour_09',
       'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23', 'Hour_nighttime']
    for col in columns_to_convert:
        data[col] = data[col].fillna(0).astype(float)

    return data

from sklearn.preprocessing import StandardScaler

def preprocess_normalize(train1, train2, train3, test1, test2, test3):
    # Selected features to normalize
    columns_to_normalize = ['tclw', 'tciw', 'sp', 'r', 'tcc', '10u', '10v', '2t', 'ssrd', 'strd', 'tsr', 'tp']
    scaler = StandardScaler()

    # Concatenate all data sets into data_total
    data_total = pd.concat([train1, train2, train3, test1, test2, test3])

    # Normalize data
    data_normalized = data_total.copy()
    data_normalized[columns_to_normalize] = scaler.fit_transform(data_total[columns_to_normalize])

    # Compute cubic features of most important features for improved performance on high derivations
    cubic_features = ['strd', 'r', 'tsr', '2t', 'sp', 'tcc']
    for feature in cubic_features:
        feature_name = f"{feature}_cubic"
        data_normalized[feature_name] = data_normalized[feature] ** 3

    data_normalized = data_normalized.drop("Day", axis=1, errors='ignore')

    # Split data_normalized again to return the original shaped data
    train1_processed = data_normalized[:len(train1)]
    train2_processed = data_normalized[len(train1):len(train1)+len(train2)]
    train3_processed = data_normalized[len(train1)+len(train2):len(train1)+len(train2)+len(train3)]
    test1_processed = data_normalized[len(train1)+len(train2)+len(train3):len(train1)+len(train2)+len(train3)+len(test1)]
    test2_processed = data_normalized[len(train1)+len(train2)+len(train3)+len(test1):len(train1)+len(train2)+len(train3)+len(test1)+len(test2)]
    test3_processed = data_normalized[len(train1)+len(train2)+len(train3)+len(test1)+len(test2):]

    return train1_processed, train2_processed, train3_processed, test1_processed, test2_processed, test3_processed

def jitter_time_series(data, noise_level=0.001):
    columns_to_add_noise = ['tclw', 'tciw', 'sp', 'r', 'tcc', '10u', '10v', '2t', 'ssrd', 'strd',
       'tsr', 'tp', 'power']

    # Identify rows corresponding to the first 4 months and extract specific parts
    mask_first_four_months = (data['Month_1'] == 1) | (data['Month_2'] == 1) | (data['Month_3'] == 1) | (data['Month_4'] == 1)
    subset = data.loc[mask_first_four_months, columns_to_add_noise].copy()
    subset_add = data.loc[mask_first_four_months].copy()

    # Apply jittering and add random noise
    noise = np.random.normal(0, noise_level, subset.shape)
    subset_jittered = subset + noise

    # Replace the noisy subset in the wider subset and add to test data
    subset_add.loc[mask_first_four_months, columns_to_add_noise] = subset_jittered
    data_augmented = pd.concat([data, subset_add])

    return data_augmented


