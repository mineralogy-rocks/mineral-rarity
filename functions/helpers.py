import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# -*- coding: utf-8 -*-

class Scaler():
    def __init__(self):
        self.scaler = StandardScaler()

    def scale_features(self, np_array):
        """Scale by StandardScale of the dataset"""

        self.scaler.fit(np_array)

        return self.scaler.transform(np_array)

    def descale_features(self, np_array):
        """ Descale by StandardScale of the dataset"""

        return self.scaler.inverse_transform(np_array)


def transform_data(df, ScalerClass):
    """ Transform data and return transformed, raw, scaled, normalized datasets """

    df = df.loc[df['locality_counts'].notna()]
    df['locality_counts'] = pd.to_numeric(df['locality_counts'])

    raw_locality_mineral_pairs = _construct_mineral_locality_pairs(df.copy())
    log_locality_mineral_pairs = _standardize_features(raw_locality_mineral_pairs.copy(), skewed_features=['mineral_count', 'locality_counts'])

    raw_locality_1d = _construct_locality_features(df.copy())
    log_locality_1d = _standardize_features(raw_locality_1d)
    scaled_locality_1d = ScalerClass.scale_features(log_locality_1d)

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


def get_discovery_rate_all(data, min_year=1500):
    """ Get all minerals counts grouped by the discovery year (all from MR)"""

    discovery_rate_all = data.loc[
        (data['discovery_year'].notna()) & (data['discovery_year'] > min_year)]

    discovery_rate_all = discovery_rate_all.sort_values('discovery_year', ascending=True)[['discovery_year']]

    discovery_rate_all = discovery_rate_all.groupby('discovery_year').agg({'discovery_year': 'count'})

    discovery_rate_all.rename(columns={'discovery_year': 'count'}, inplace=True)

    return discovery_rate_all


def get_discovery_rate_endemic(data):
    """ Get endemic minerals counts grouped by the discovery year """

    discovery_rate_endemic = data.loc[
        (data['locality_counts'] == 1) & (data['discovery_year'].notna())]

    discovery_rate_endemic = discovery_rate_endemic.sort_values('discovery_year', ascending=True)

    discovery_rate_endemic = discovery_rate_endemic.groupby('discovery_year').agg({'locality_counts': 'sum'})

    discovery_rate_endemic.rename(columns={'locality_counts': 'count'}, inplace=True)

    return discovery_rate_endemic