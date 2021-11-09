import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Violinplot of 0.25, 0.5 and 0.75 quantiles

plt.violinplot(locs['locality_counts'], points=200, vert=False, widths=0.1,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.25, 0.5, 0.75], bw_method=2)

plt.title('Empirical distribution of raw localities data', fontsize=10)

plt.savefig(f"figures/quantiles/violin_plot.jpeg", dpi=300, format='jpeg')

plt.close()
