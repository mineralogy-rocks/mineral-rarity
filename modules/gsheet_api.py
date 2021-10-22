import asyncio
import gspread_asyncio
import gspread
import pandas as pd
import numpy as np

from google.oauth2.service_account import Credentials


class GsheetApi():

    def __init__(self):

        self.agcm = gspread_asyncio.AsyncioGspreadClientManager
        self.sheet_mapping = [
            {'ws_name': 'Masterlist2', 'ss_name': 'Status data', 'local_name': 'status_data'},
            {'ws_name': 'Masterlist2', 'ss_name': 'Nickel-Strunz', 'local_name': 'nickel_strunz'},
            {'ws_name': 'Groups', 'ss_name': 'Groups_ver1', 'local_name': 'groups_formulas'},
            {'ws_name': 'Masterlist2', 'ss_name': 'Names data', 'local_name': 'names'},
            {'ws_name': 'Locality_count_rruff', 'ss_name': 'loc_2019', 'local_name': 'loc_2019'},
            {'ws_name': 'Locality_count_rruff', 'ss_name': 'loc_2020', 'local_name': 'loc_2020'},
            {'ws_name': 'Locality_count_rruff', 'ss_name': 'loc_2021', 'local_name': 'loc_2021'},
        ]
        self.status_data = None
        self.nickel_strunz = None
        self.groups_formulas = None
        self.names = None

        self.loc_2018 = None
        self.loc_2019 = None
        self.loc_2020 = None


    def get_local_name(self, ss_name):
        '''
        A function which returns local_name from self.sheet_mappings
        '''
        try:
            return [sheet['local_name'] for sheet in self.sheet_mapping if sheet['ss_name'] == ss_name]
        except:
            print(f'ss_name="{ss_name}" is not present in gsheets_api!')


    def get_creds(self):

        creds = Credentials.from_service_account_file('gsheets_credentials.json')
        scoped = creds.with_scopes([
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ])
        return scoped


    async def as_get_sheet_data(self, worksheet_name: str, sheet_name: str):
        """
        a function to upload gsheet data into pandas df
        Args:
            worksheet_name: name of worksheet
            sheet_name: name of sheet
        Returns:
            pandas dataframe
        """

        agc = await self.agcm(self.get_creds).authorize()
        try:
            print(f'started grabbing data of {sheet_name}')
            ags = await agc.open(worksheet_name)
            agw = await ags.worksheet(sheet_name)
            data = await agw.get_all_values()
            print(f'Processed {sheet_name}')
            headers = data.pop(0)
            output = pd.DataFrame(data, columns=headers).replace(r'', np.nan)
            local_var = self.get_local_name(ss_name=sheet_name)[0]
            setattr(self, local_var, output)
            return output
        except gspread.exceptions.GSpreadException as error:
            print(error)
            print(f'An error occurred while reading sheet_name={sheet_name}')


    async def main(self):

        await asyncio.gather(*(self.as_get_sheet_data(sheet['ws_name'], sheet['ss_name']) for sheet in
                               self.sheet_mapping))


    def run_main(self):

        import time
        s = time.perf_counter()
        asyncio.run(self.main())
        elapsed = time.perf_counter() - s
        print(f"Executed in {elapsed:0.2f} seconds.")