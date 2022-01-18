from modules.gsheet_api import GsheetApi
import pandas as pd

# -*- coding: utf-8 -*-


GsheetApi = GsheetApi()
GsheetApi.run_main()

# Trace history of mineral localities changes

ima_minerals = GsheetApi.status_data.loc[GsheetApi.status_data['all_indexes'].str.match('0.0')].reset_index(drop=True)[['Mineral_Name']]

ima_minerals.rename(columns={'Mineral_Name': 'mineral_name'}, inplace=True)

ima_minerals.set_index('mineral_name', inplace=True)


loc_2019 = GsheetApi.loc_2019.loc[(GsheetApi.loc_2019['count_locations'].notna()) & (GsheetApi.loc_2019['mineral_name'].notna())][['mineral_name', 'count_locations']].set_index('mineral_name')
loc_2020 = GsheetApi.loc_2020.loc[(GsheetApi.loc_2020['count_locations'].notna()) & (GsheetApi.loc_2020['mineral_name'].notna())][['mineral_name', 'count_locations']].set_index('mineral_name')
loc_2021 = GsheetApi.loc_2021.loc[(GsheetApi.loc_2021['count_locations'].notna()) & (GsheetApi.loc_2021['mineral_name'].notna())][['mineral_name', 'count_locations']].set_index('mineral_name')


locs = loc_2019.join(loc_2020, how='outer', lsuffix='_2019', rsuffix='_2020').join(loc_2021, how='outer')

locs.rename(columns={'count_locations': 'count_locations_2020'}, inplace=True)

locs['count_locations_2019'] = pd.to_numeric(locs['count_locations_2019'])
locs['count_locations_2020'] = pd.to_numeric(locs['count_locations_2020'])
locs['count_locations_2021'] = pd.to_numeric(locs['count_locations_2021'])

# Find minerals which transferred from one rarity state to another

locs = locs.loc[locs['count_locations_2019'].notna() & locs['count_locations_2020'].notna() & locs['count_locations_2021'].notna()]
locs_changed = locs.loc[
    (locs['count_locations_2019'] != locs['count_locations_2020']) | (locs['count_locations_2020'] != locs['count_locations_2021'])
]


# Find endemic minerals

locs.loc[locs['count_locations_2020'] == 1]


# Subset to only IMA-approved minerals

ima_locs = locs.join(ima_minerals, how='inner')