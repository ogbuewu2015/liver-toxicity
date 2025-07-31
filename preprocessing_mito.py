# preprocessing.py
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def remove_low_variance(input_data, threshold=0.1):
    selector = VarianceThreshold(threshold)
    selector.fit(input_data)
    retained_features = input_data.columns[selector.get_support(indices=True)]
    return input_data[retained_features], retained_features, selector

def remove_correlated_features(descriptors, threshold=0.9):
    correlated_matrix = descriptors.corr().abs()
    upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    descriptors_correlated_dropped = descriptors.drop(columns=to_drop, axis=1)
    return descriptors_correlated_dropped, to_drop

def scale_training_data(train_data):
    float_columns = train_data.select_dtypes(include=['float']).columns
    non_float_columns = train_data.drop(columns=float_columns)

    scaler = StandardScaler()
    scaled_floats = pd.DataFrame(
        scaler.fit_transform(train_data[float_columns]),
        columns=float_columns,
        index=train_data.index
    )

    scaled_train = pd.concat([scaled_floats, non_float_columns], axis=1)
    return scaled_train, scaler, float_columns

def scale_new_data(new_data, scaler, float_columns):
    float_data = new_data[float_columns]
    non_float_columns = new_data.drop(columns=float_columns)

    scaled_floats = pd.DataFrame(
        scaler.transform(float_data),
        columns=float_columns,
        index=new_data.index
    )

    scaled_data = pd.concat([scaled_floats, non_float_columns], axis=1)
    return scaled_data
