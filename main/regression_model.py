import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from functions.helpers import parse_mindat, parse_rruff, get_discovery_rate_all, get_discovery_rate_endemic, get_endemic_proportion

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for training a Regression model on discovery rates of all and endemic minerals
"""

locs_md = pd.read_csv('data/mindat_locs.csv', sep=',')
rruff_data = pd.read_csv('data/RRUFF_Export.csv', sep=',')

locs_md = parse_mindat(locs_md)
rruff_data = parse_rruff(rruff_data)


mindat_rruff = locs_md.join(rruff_data, how='outer')
mindat_rruff = mindat_rruff[['discovery_year', 'locality_counts']]

## Get discovery rates (localities from mindat, discovery year from RRUFF)

discovery_rate_all = get_discovery_rate_all(mindat_rruff)
discovery_rate_endemic = get_discovery_rate_endemic(mindat_rruff)
endemic_proportion = get_endemic_proportion(discovery_rate_endemic, discovery_rate_all)

# Create a model of before 2012, use without outliers and calculate True endemicity
# Here X (attribute) is discovery year and Y (label) is number of endemic minerals discovered during that year

X = np.array(discovery_rate_endemic.loc[discovery_rate_endemic.index < 2012].index).reshape(-1,1)
y = discovery_rate_endemic.loc[discovery_rate_endemic.index < 2012, 'count'].to_numpy(dtype=int).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Ridge(alpha=0.001, fit_intercept=True)

model.fit(X_train, y_train)

model.predict(X_test)