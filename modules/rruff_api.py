import pandas as pd
import asyncio


class RruffApi():


    def __init__(self):

        self.base_url = 'https://rruff.info/mineral_list/MED/exporting/'
        self.sheet_mapping = [
            {'url': '2016_01_15_data/tbl_locality_age_cache_alt.csv', 'name': 'loc_2016'},
            {'url': '2017_06_22_data/tbl_locality_age_cache_alt.csv', 'name': 'loc_2017'},
            {'url': '2018_03_27_data/tbl_locality_age_cache_alt.csv', 'name': 'loc_2018'},
            {'url': '2019_05_22/tbl_locality_age_cache_alt.csv', 'name': 'loc_2019'},
        ]
        self.status_data = None
        self.nickel_strunz = None
        self.groups_formulas = None
        self.names = None

        self.loc_2016 = None
        self.loc_2017 = None
        self.loc_2018 = None
        self.loc_2019 = None


    async def read_locality_data(self, details: dict):
        """
        read locs from RRUFF
        Args:
            details: dict with 'url' and 'name', which is a local name of a dataframe
        Returns:
            pandas dataframe
        """

        try:

            print(f'started grabbing data from {self.base_url + details["url"]}')

            data = await pd.read_csv(f'{self.base_url + details["url"]}', delimiter='\t')

            print(f'Processed {details["name"]}')

            setattr(self, details['name'], data)

            return data

        except Exception as error:

            print(f'An error occurred while retrieving data from {self.base_url + details["url"]}: {error}')


    async def main(self):

        await asyncio.gather(*(self.as_get_sheet_data(sheet['ws_name'], sheet['ss_name']) for sheet in
                               self.sheet_mapping))


    def run_main(self):

        import time
        s = time.perf_counter()
        asyncio.run(self.main())
        elapsed = time.perf_counter() - s
        print(f"Executed in {elapsed:0.2f} seconds.")

# import raw data

loc_2016 = pd.read_csv('https://rruff.info/mineral_list/MED/exporting/2016_01_15_data/tbl_locality_age_cache_alt.csv', delimiter='\t')

loc_2017 = pd.read_csv('https://rruff.info/mineral_list/MED/exporting/2017_06_22_data/tbl_locality_age_cache_alt.csv', delimiter='\t')

loc_2018 = pd.read_csv('https://rruff.info/mineral_list/MED/exporting/2018_03_27_data/tbl_locality_age_cache_alt.csv', delimiter='\t')

loc_2019 = pd.read_csv('https://rruff.info/mineral_list/MED/exporting/2019_05_22/tbl_locality_age_cache_alt.csv', delimiter='\t')


# create history of locality counts changes

locs = pd.DataFrame()

for loc in ['2016', '2017', '2018', '2019']:

    loc = loc.loc[loc['at_locality'] == 1]
    loc = loc.groupby(['mineral_display_name']).agg(
        locality_counts=pd.NamedAgg(column="mindat_id", aggfunc="count")
    )
    locs

loc_2016 = loc_2016.loc[loc_2016['at_locality'] == 1]
loc_2016_counts = loc_2016.groupby(['mineral_display_name']).agg(
    locality_counts=pd.NamedAgg(column="mindat_id", aggfunc="count")
)