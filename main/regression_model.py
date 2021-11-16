import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Plot endemic proportion (endemic count / all count * 100%)

plt.scatter(endemic_proportion.index, endemic_proportion['proportion'], color='#5FD6D1', marker='o', s=20,
            edgecolors='black', linewidths=0.1)

plt.xlabel('Discovery year')
plt.ylabel('Number of endemic minerals / Number of all minerals')
plt.title('The rate of endemic minerals discovery')

plt.savefig(f"figures/endemic_minerals/proportion.jpeg", dpi=300, format='jpeg')

plt.close()


# Create a model of before 2012, use without outliers and calculate True endemicity
# Here X (attribute) is discovery year and Y (label) is number of endemic minerals discovered during that year

transformer = PolynomialFeatures(degree=2, include_bias=False)

X_all = endemic_proportion[['count_all']]
X_all.reset_index(inplace=True)
X_all = np.array(X_all[['discovery_year']], dtype=int)
X = X_all[(X_all > 1900) & (X_all < 2012)].reshape(-1,1)
X_after_1900 = X_all[(X_all > 1900)].reshape(-1,1)

X_ = transformer.fit_transform(X)
X_after_1900_ = transformer.fit_transform(X_after_1900)

y_all = endemic_proportion.reset_index().to_numpy(dtype=int)
y = y_all[(y_all[:, 0] > 1900) & (y_all[:, 0] < 2012)][:, 1]

X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=0, shuffle=False)

model = LinearRegression(fit_intercept=True)

model.fit(X_, y)

model.score(X_, y)

predicted_after_1900 = model.predict(X_after_1900_)

# Plot the model
plt.scatter(y_all[(y_all[:, 0] > 1900)][:, 0], y_all[(y_all[:, 0] > 1900)][:, 1], color='green', marker='o', s=20,
            edgecolors='black', linewidths=0.1)
plt.plot(X_after_1900, predicted_after_1900, color="blue", linewidth=1)

plt.savefig(f"figures/endemic_minerals/linear_regression.jpeg", dpi=300, format='jpeg')

plt.close()