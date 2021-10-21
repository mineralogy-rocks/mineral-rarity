from modules.gsheet_api import gsheet_api
import pandas as pd

gsheet_api = gsheet_api()


gsheet_api.run_main()

# Trace history of mineral localities changes

ima_minerals = gsheet_api.status_data.loc[gsheet_api.status_data['all_indexes'].str.match('0.0')].reset_index(drop=True)[['Mineral_Name']]

ima_minerals.rename(columns={'Mineral_Name': 'mineral_name'}, inplace=True)


loc_2018 = gsheet_api.loc_2018.loc[(gsheet_api.loc_2018['count_locations'].notna()) & (gsheet_api.loc_2018['mineral_name'].notna())][['mineral_name', 'count_locations']].set_index('mineral_name')
loc_2019 = gsheet_api.loc_2019.loc[(gsheet_api.loc_2019['count_locations'].notna()) & (gsheet_api.loc_2019['mineral_name'].notna())][['mineral_name', 'count_locations']].set_index('mineral_name')
loc_2020 = gsheet_api.loc_2020.loc[(gsheet_api.loc_2020['count_locations'].notna()) & (gsheet_api.loc_2020['mineral_name'].notna())][['mineral_name', 'count_locations']].set_index('mineral_name')


locs = loc_2018.join(loc_2019, how='outer', lsuffix='_2018', rsuffix='_2019').join(loc_2020, how='outer')

locs.rename(columns={'count_locations': 'count_locations_2020'}, inplace=True)

locs['count_locations_2018'] = pd.to_numeric(locs['count_locations_2018'])
locs['count_locations_2019'] = pd.to_numeric(locs['count_locations_2019'])
locs['count_locations_2020']= pd.to_numeric(locs['count_locations_2020'])

locs = locs.loc[locs['count_locations_2018'].notna() & locs['count_locations_2019'].notna() & locs['count_locations_2020'].notna()]

# Find minerals which transferred from one rarity state to another

locs_changed = locs.loc[
    (locs['count_locations_2018'] != locs['count_locations_2019']) | (locs['count_locations_2019'] != locs['count_locations_2020'])
]

locs.loc[locs['count_locations_2019'] == 1]