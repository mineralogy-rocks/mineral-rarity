import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functions.helpers import parse_mindat, parse_rruff, get_discovery_rate_all, get_discovery_rate_endemic, \
    get_endemic_proportion

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Code for training two Regression models on discovery rates of endemic minerals between 1900-2012 and 1900-2021 
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
endemic_proportion.reset_index(inplace=True)

# Plot endemic proportion (endemic count / all count * 100%)

plt.scatter(endemic_proportion.index, endemic_proportion['proportion'], color='#5FD6D1', marker='o', s=20,
            edgecolors='black', linewidths=0.1)

plt.xlabel('Discovery year')
plt.ylabel('Number of endemic minerals / Number of all minerals')
plt.title('The rate of endemic minerals discovery')

plt.savefig(f"figures/endemic_minerals/proportion.jpeg", dpi=300, format='jpeg')

plt.close()



# Create different linear models for a subsets of years
# Here X (attribute) is discovery year and Y (label) is number of endemic minerals discovered during that year

transformer = PolynomialFeatures(degree=3, include_bias=False)
post_1900 = endemic_proportion.loc[(endemic_proportion['discovery_year'] > 1900)]

X = {
    '1900_2021': post_1900[['discovery_year']].to_numpy(dtype=int).reshape(-1, 1),
}

# polynomial transformed
X_ = {
    '1900_2021': transformer.fit_transform(X['1900_2021']),
}

y = {
    '1900_2021': post_1900[['count_endemic']].to_numpy(dtype=int)
}

# Plot initial discovery rate of endemic minerals
plt.scatter(post_1900[['discovery_year']], post_1900[['count_endemic']], color='green', marker='o', s=20,
            edgecolors='black', linewidths=0.1)

training_sets = [
    { 'year': 1990, 'color': 'green', 'linestyle': 'dotted' },
    { 'year': 2000, 'color': 'magenta', 'linestyle': 'dotted' },
    { 'year': 2011, 'color': 'blue', 'linestyle': 'dashed' },
    { 'year': 2021, 'color': 'olive', 'linestyle': 'dashed' }
]

for training_set in training_sets:

    pre_year = post_1900.loc[post_1900['discovery_year'] <= training_set['year']]

    X['pre_year'] = pre_year[['discovery_year']].to_numpy(dtype=int).reshape(-1, 1)

    # polynomial transformed
    X_['pre_year'] = transformer.fit_transform(X['pre_year'])
    y['pre_year'] = pre_year[['count_endemic']].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(X_['pre_year'], y['pre_year'], test_size = 0.1, random_state = 53, # 51 and 52
                                                        shuffle=True)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    # model.fit(X_['pre_year'], y['pre_year'])

    y_pred_all = model.predict(X_['1900_2021'])
    y_pred = model.predict(X_test)

    # Calculate RMSE and r2
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)
    exp_variance = explained_variance_score(y_test, y_pred)

    print(f'1900-{training_set["year"]} range \n'
          f'R2 = {R2};\n'
          f'RMSE = {RMSE}; \n'
          f'Coefficients: {model.coef_}; \n'
          f'Explained variance: {exp_variance} \n'
          f'----------------------------------')

    plt.plot(post_1900[['discovery_year']], y_pred_all, color=training_set['color'], linestyle=training_set['linestyle'], linewidth=1)

plt.savefig(f"figures/endemic_minerals/linear_regression_3_polynomial.jpeg", dpi=300, format='jpeg')

plt.close()