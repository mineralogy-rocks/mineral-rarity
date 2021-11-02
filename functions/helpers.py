import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# -*- coding: utf-8 -*-


def transform_data(df):
    # for further studying the results, you will need original data
    df_original = df.copy()

    df_transformed = _construct_features(df)
    df_transformed = _standardize_features(df_transformed)
    df_transformed = _scale_features(df_transformed)

    return df_original, df_transformed


def _construct_features(df):
    # adding new features that better represent the problem
    # you need to have domain knowledge

    # example of creating a new function based on two existing ones from real estate domain
    # - cost of buying a house is 6% of the house price
    # - standard deposit is 10% of price
    df['first_payment'] = df['house_price'] * 0.06 + df['house_price'] * 0.1

    # anuitete formula https://myfin.by/wiki/term/annuitetnyj-platyozh
    n = 360         # loan for 30 years
    i = 0.00283     # loan rate 3.4%
    k = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    df['monthly_payment'] = (df['house_price'] - df['house_price'] * 0.1) * k

    return df


def _standardize_features(df: pd.DataFrame, skewed_features=[]):
    """Log transform of the df"""

    for feature in skewed_features:
        df[feature] = np.log(df[feature])

    return df


def _scale_features(df, features_to_scale=[]):
    """StandardScale of the dataset"""

    scaler = StandardScaler()

    for feature in features_to_scale:
        df[[feature]] = scaler.fit_transform(df[[feature]])

    return df