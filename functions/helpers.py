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


def parse_rruff(data):
    """ Clean and transform RRUFF data  """

    data.loc[data['Year First Published'] == 0, 'Year First Published'] = np.nan
    data.rename(columns={'Mineral Name': 'mineral_name', 'Year First Published': 'discovery_year'}, inplace=True)

    rruff_data = data[['mineral_name', 'discovery_year']]
    rruff_data.set_index('mineral_name', inplace=True)

    return rruff_data


def parse_mindat(data):
    """ Clean and transform mindat data  """

    data.replace('\\N', np.nan, inplace=True)

    data.rename(columns={'loccount': 'locality_counts'}, inplace=True)

    ## Don't have any clue why mindat keeps `0` in imayear column and mixes dtypes in others (?!)

    data['imayear'].replace('0', np.nan, inplace=True)

    locs_md = data[['name', 'imayear', 'yeardiscovery', 'yearrruff', 'locality_counts']]
    locs_md.loc[:, 'imayear'] = pd.to_numeric(locs_md['imayear'])
    locs_md.loc[:, 'yearrruff'] = pd.to_numeric(locs_md['yearrruff'])

    locs_md.loc[:, 'locality_counts'] = pd.to_numeric(locs_md['locality_counts'])
    locs_md.loc[~locs_md['yeardiscovery'].str.match(r'[0-9]{4}', na=False), 'yeardiscovery'] = np.nan
    locs_md.loc[:, 'yeardiscovery'] = pd.to_numeric(locs_md['yeardiscovery'])

    locs_md.set_index('name', inplace=True)

    return locs_md

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


def get_endemic_proportion(discovery_rate_endemic, discovery_rate_all):
    """ Calculate the proportion of endemic minerals from all """

    endemic_all_prop = discovery_rate_endemic.join(discovery_rate_all, how='inner', lsuffix='_endemic', rsuffix='_all')

    endemic_all_prop['proportion'] = endemic_all_prop['count_endemic'] / endemic_all_prop['count_all'] * 100

    return endemic_all_prop

def calculate_cooccurrence_matrix(data0, data1, norm=False):
    cooccurrence = pd.crosstab(data0, data1, normalize=norm)
    np.fill_diagonal(cooccurrence.values, 0)
    return cooccurrence