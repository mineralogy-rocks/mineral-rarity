import io

import asyncio
import aiohttp

import pandas as pd


# -*- coding: utf-8 -*-

class RruffApi():


    def __init__(self):

        self.base_url = 'https://rruff.info/mineral_list/MED/exporting/'
        self.url_mapping = [
            {'url': '2016_01_15_data', 'year': '2016'},
            {'url': '2017_06_22_data', 'year': '2017'},
            {'url': '2018_03_27_data', 'year': '2018'},
            {'url': '2019_05_22', 'year': '2019'},
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
            
            async with client.get(f'{self.base_url + details["url"] + "/tbl_locality_age_cache.csv"}', headers={'Connection': 'keep-alive'}) as response:

                with io.BytesIO(await response.content.read()) as file:

                    file.seek(0)

                    _loc_min = pd.read_csv(file, delimiter='\t')

                    # data = self.process_initial_locs(data, details['year'])


            async with client.get(f'{self.base_url + details["url"] + "/tbl_locality.csv"}', headers={'Connection': 'keep-alive'}) as response:

                with io.BytesIO(await response.content.read()) as file:

                    file.seek(0)

                    _loc = pd.read_csv(file, delimiter='\t')


            async with client.get(f'{self.base_url + details["url"] + "/tbl_mineral.csv"}',
                                  headers={'Connection': 'keep-alive'}) as response:

                with io.BytesIO(await response.content.read()) as file:
                    file.seek(0)

                    _min = pd.read_csv(file, delimiter='\t')


            data = self.process_initial_locs(_loc_min, _loc, _min, details['year'])

            setattr(self, f'_loc_{details["year"]}', data)

        except Exception as error:

            print(f'An error occurred while retrieving data from {self.base_url + details["url"]}: {error}')


    def process_initial_locs(self, loc_min, loc, min, year):
        """
        Process initial data uploaded from RRUFF
        :param
            loc_min: pandas dataframe with loc and min ids
            loc: pandas dataframe with loc ids
            min: pandas dataframe with min ids
        :return:
            processed dataframe, grouped by mineral_name
        """

        loc = loc.loc[(loc['is_bottom_level'] == 1) & (loc['is_meteorite'] == 0)][['locality_id']]

        loc['locality_id'] = pd.to_numeric(loc['locality_id'])
        loc_min['locality_id'] = pd.to_numeric(loc_min['locality_id'])

        loc_min = loc_min[['locality_id', 'mineral_id']]

        locs = loc.set_index('locality_id').join(loc_min.set_index('locality_id'), how='inner')

        mins = locs.set_index('mineral_id').join(min.set_index('mineral_id'), how='inner')


        mins = mins.groupby('mineral_name').agg(
            locality_counts=pd.NamedAgg(column="mineral_name", aggfunc="count")
        ).sort_values(by='locality_counts')

        mins['locality_counts'] = pd.to_numeric(mins['locality_counts'])

        mins.rename(columns={'locality_counts': f'locality_counts_{year}'}, inplace=True)

        return mins


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