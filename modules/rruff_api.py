import io

import asyncio
import aiohttp

import pandas as pd


# -*- coding: utf-8 -*-

class RruffApi():


    def __init__(self):

        self.base_url = 'https://rruff.info/mineral_list/MED/exporting/'
        self.url_mapping = [
            {'url': '2016_01_15_data/tbl_locality_age_cache_alt.csv', 'year': '2016'},
            {'url': '2017_06_22_data/tbl_locality_age_cache_alt.csv', 'year': '2017'},
            {'url': '2018_03_27_data/tbl_locality_age_cache_alt.csv', 'year': '2018'},
            {'url': '2019_05_22/tbl_locality_age_cache_alt.csv', 'year': '2019'},
        ]

        self._loc_2016 = None
        self._loc_2017 = None
        self._loc_2018 = None
        self._loc_2019 = None

        self.locs = pd.DataFrame()


    async def read_locality_data(self, client, details: dict):
        """
        read locs from RRUFF
        Args:
            details: dict with 'url' and 'name', which is a local name of a dataframe
        Returns:
            pandas dataframe
        """

        print(f'Start uploading {details["year"]}...')

        try:
            
            async with client.get(f'{self.base_url + details["url"]}', headers={'Connection': 'keep-alive'}) as response:

                with io.BytesIO(await response.content.read()) as file:

                    file.seek(0)

                    data = pd.read_csv(file, delimiter='\t')

                    if details['year'] != 2019:

                        data = self.process_initial_locs(data, details['year'])

                    setattr(self, f"_loc_{details['year']}", data)

                    return data

        except Exception as error:

            print(f'An error occurred while retrieving data from {self.base_url + details["url"]}: {error}')


    def process_initial_locs(self, df, year):
        """
        Process initial data uploaded from RRUFF
        :param df: pandas dataframe
        :return: processed dataframe, grouped by mindat_id and localities counts
        """

        df = df.loc[df['at_locality'] == 1]

        df = df.groupby(['mineral_display_name']).agg(
                locality_counts=pd.NamedAgg(column="mindat_id", aggfunc="count")
            )

        df['locality_counts'] = pd.to_numeric(df['locality_counts'])

        df.rename(columns={'locality_counts': f'locality_counts_{year}'}, inplace=True)

        return df


    def join_locs(self):
        """
        Join all dfs and create a subset for comparison
        :return: set df to self.locs
        """

        for url in self.url_mapping:

            current_loc = getattr(self, f'_loc_{url["year"]}')

            if len(self.locs):
                self.locs = self.locs.join(current_loc, how='outer')

            else:
                self.locs = current_loc


    async def main(self):

        async with aiohttp.ClientSession() as client:

            futures = [self.read_locality_data(client, url) for url in self.url_mapping]

            return await asyncio.gather(*futures)


    def run_main(self):

        import time
        s = time.perf_counter()

        asyncio.run(self.main())

        self.join_locs()

        elapsed = time.perf_counter() - s
        print(f"Executed in {elapsed:0.2f} seconds.")



    # test localities from other folder

    _loc_min = pd.read_csv('https://rruff.info/mineral_list/MED/exporting/2016_01_15_data/tbl_locality_age_cache.csv', delimiter='\t')
    _loc = pd.read_csv('https://rruff.info/mineral_list/MED/exporting/2016_01_15_data/tbl_locality.csv',delimiter='\t')
    _min = pd.read_csv('https://rruff.info/mineral_list/MED/exporting/2016_01_15_data/tbl_mineral.csv',delimiter='\t')

    _loc = _loc.loc[(_loc['is_bottom_level'] == 1) & (_loc['is_meteorite'] == 0)][['locality_id']].set_index('locality_id')

    _loc_min = _loc_min[['locality_id', 'mineral_id']]
    _loc_min = _loc_min.set_index('locality_id')

    locs = _loc_min.join(_loc, how='inner').reset_index(drop=True).set_index('mineral_id')


    mins = locs.join(_min, how='inner')


    mins.loc[(mins['mineral_name'] == 'Quartz')]
