from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import copy
import pandas as pd

def preprocess_data(data, columns_to_exclude_normalize=None):
    if columns_to_exclude_normalize is None:
        columns_to_exclude_normalize = []
    
    # Extract features from the "datetime" column
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['Hour'] = data.index.hour

    # Create a single month category column for different days
    data['month_cat'] = np.select(
        [(data['Day'] <= 10), (data['Day'] > 10) & (data['Day'] <= 20), (data['Day'] > 20)],
        ['beginning', 'middle', 'end'],
        default='unknown_category'
    )

    # Perform one-hot encoding for month and days
    data_encoded = pd.get_dummies(data, columns=['Month', 'month_cat'])

    # Specify specific labels for Hour
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

    # Identify columns to normalize
    columns_to_normalize = [col for col in data_encoded.columns if col not in columns_to_exclude_normalize]

    # Normalize selected features
    scaler = StandardScaler()
    data_normalized = data_encoded.copy()
    data_normalized[columns_to_normalize] = scaler.fit_transform(data_encoded[columns_to_normalize])

    # Compute cubic features and add them to the DataFrame
    cubic_features = ['strd', 'r', 'tsr', '2t', 'sp', 'tcc']
    for feature in cubic_features:
        feature_name = f"{feature}_cubic"
        data_normalized[feature_name] = data_normalized[feature] ** 3

    data_normalized = data_normalized.drop("Day", axis=1, errors='ignore')

    return data_normalized