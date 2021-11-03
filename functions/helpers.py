import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# -*- coding: utf-8 -*-


def transform_data(df):
    """ Transform data and return original df too """
    df_original = df.copy()

    df_transformed = _construct_locality_features(df)
    df_transformed_non_scaled = df_transformed.copy()
    df_transformed = _standardize_features(df_transformed, skewed_features=['mineral_count', 'locality_counts'])
    df_transformed = _scale_features(df_transformed, features_to_scale=['mineral_count', 'locality_counts'])

    return df_original, df_transformed_non_scaled, df_transformed


def _construct_locality_features(df):
    """ Create initial df with all features transformed """

    df = df.loc[df['locality_counts'].notna()][['locality_counts']]
    df['locality_counts'] = pd.to_numeric(df['locality_counts'])

    df['mineral_count'] = 1

    df = df.groupby(by='locality_counts').agg(mineral_count=pd.NamedAgg(column="mineral_count", aggfunc="sum"))
    df = df.sort_values(by=['mineral_count', 'locality_counts'], ascending=(False, True), axis=0).reset_index(
        level=0)

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