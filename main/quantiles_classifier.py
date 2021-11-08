import pandas as pd
import numpy as np

from modules.gsheet_api import GsheetApi

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Functions for classifying mineral locality counts into clusters/categories
using k-means clustering
"""


GsheetApi = GsheetApi()
GsheetApi.run_main()

locs = GsheetApi.locs.copy()

locs['locality_counts'] = pd.to_numeric(locs['locality_counts'])


# Quantiles of raw localities data

len(locs.loc[locs['locality_counts'] <= 4]) / len(locs) * 100 # Rare minerals volume
len(locs.loc[locs['locality_counts'] > 18]) / len(locs) * 100 # Ubiquitous minerals volume
len(locs.loc[(locs['locality_counts'] > 4) & (locs['locality_counts'] <= 18)]) / len(locs) * 100 # Transitional minerals volume

for q in [0.25, 0.5, 0.75]:
    print(f"{q} quantile: - {np.quantile(locs['locality_counts'].to_numpy(dtype=int), q=q)}")