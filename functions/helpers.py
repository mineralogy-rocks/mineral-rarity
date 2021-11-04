import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# -*- coding: utf-8 -*-


def transform_data(df):
    """ Transform data and return transformed, raw, scaled, normalized datasets """

    df = df.loc[df['locality_counts'].notna()]
    df['locality_counts'] = pd.to_numeric(df['locality_counts'])

    raw_locality_mineral_pairs = _construct_mineral_locality_pairs(df)
    log_locality_mineral_pairs = _standardize_features(raw_locality_mineral_pairs.copy(), skewed_features=['mineral_count', 'locality_counts'])

    raw_locality_1d = _construct_locality_features(df)
    log_locality_1d = _standardize_features(raw_locality_1d)
    scaled_locality_1d = _scale_features(log_locality_1d)

    return raw_locality_mineral_pairs, log_locality_mineral_pairs, scaled_locality_1d


def _construct_mineral_locality_pairs(df):
    """ Create dataframe grouped by locality counts
    Returns:
        df['locality_count', 'mineral_count']
    """

    df['mineral_count'] = 1

    df = df.groupby(by='locality_counts').agg(mineral_count=pd.NamedAgg(column="mineral_count", aggfunc="sum"))
    df = df.sort_values(by=['mineral_count', 'locality_counts'], ascending=(False, True), axis=0).reset_index(
        level=0)

    return df


def _construct_locality_features(df):
    """ Create numpy array with localities features in (-1,1) shape, eg [ [1], [2], [3] ] """

    df = df['locality_counts'].to_numpy(dtype=int).reshape(-1, 1)

    return df


def _standardize_features(data, skewed_features=[]):
    """ Log transform of the df or numpy array """

    if isinstance(data, pd.DataFrame):
        for feature in skewed_features:
            data[feature] = np.log(data[feature])
    else:
        data = np.log(data)

    return data


def _scale_features(np_array):
    """StandardScale of the dataset"""

    scaler = StandardScaler()

    return scaler.fit_transform(np_array)