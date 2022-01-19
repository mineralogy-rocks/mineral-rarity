import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
import seaborn as sns

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


# Use seaborn to compile regression plots


transformer = PolynomialFeatures(degree=2, include_bias=False)
post_1900 = endemic_proportion.loc[(endemic_proportion['discovery_year'] > 1900)]
all = pd.DataFrame(columns=['x', 'y', 'features_range', 'features_all', 'type'])

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


training_sets = [
    { 'year': 1990, 'color': 'green', 'linestyle': 'dotted' },
    { 'year': 2000, 'color': 'magenta', 'linestyle': 'dotted' },
    { 'year': 2011, 'color': 'blue', 'linestyle': 'dashed' },
    { 'year': 2021, 'color': 'olive', 'linestyle': 'dashed' }
]

for index, training_set in enumerate(training_sets):

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
          f'Intercept: {model.intercept_} \n'
          f'Coefficients: {model.coef_}; \n'
          f'Explained variance: {exp_variance} \n'
          f'----------------------------------')

    data_ = post_1900[['discovery_year']].rename(columns={'discovery_year': 'x'})
    data_['y'] = y_pred_all
    data_['features_range'] = post_1900.loc[post_1900['discovery_year'] <= training_set['year'], 'count_endemic']
    data_['features_all'] = post_1900[['count_endemic']]
    data_['type'] = '1900 - {year}'.format(year=training_set['year'])
    all = pd.concat([all, data_])


sns.set_theme(palette=None, style={ 'figure.facecolor': 'white', 'xtick.bottom': True, 'ytick.left': True })
plt.rcParams['axes.titlepad'] = -14

g = sns.FacetGrid(all, col="type", col_wrap=2, height=4, aspect=1, margin_titles=True, despine=False, legend_out=True)
g.map(sns.scatterplot, "x", "features_all", color='green', marker='o', s=80, edgecolor='black', linewidth=1,
      alpha=0.5)
g.map(sns.scatterplot, "x", "features_range", color='teal', marker='o', s=80, alpha=0.8, edgecolor='black', linewidth=1)
g.map(sns.regplot, "x", "features_range", order=2, truncate=False, scatter=False, ci=95, color='dodgerblue', line_kws= {'alpha':0.6, 'linewidth': 1})
g.map(sns.regplot, "x", "features_range", order=3, truncate=False, scatter=False, ci=95, color='orange', line_kws= {'alpha':0.6, 'linewidth': 1})

g.set_axis_labels("Discovery year", "Minerals count")
g.set_titles(col_template="{col_name}")
g.tight_layout()
g.add_legend(labels=['2nd degree polynomial regression', '3rd degree polynomial regression',  'features', 'training set'])
g.figure.subplots_adjust(wspace=0, hspace=0)
g.set(ylim=(-10, 100))

plt.xlabel('Discovery year')
plt.ylabel('Minerals count')

plt.savefig(f"figures/endemic_minerals/linear_regression.jpeg", dpi=300, format='jpeg')

plt.close()



# Predict number of "true" endemic species for 2001-2021 using the model trained on 1900-2000 span
# with a 2-degree polynomial transform

post_1900 = endemic_proportion.loc[(endemic_proportion['discovery_year'] > 1900)]
pre_year = post_1900.loc[post_1900['discovery_year'] <= 2011]

X = {
    '1900_2021': post_1900[['discovery_year']].to_numpy(dtype=int).reshape(-1, 1),
}

y = {
    '1900_2021': post_1900[['count_endemic']].to_numpy(dtype=int)
}

transformer = PolynomialFeatures(degree=2, include_bias=False)

X_ = {
    '1900_2021': transformer.fit_transform(X['1900_2021']),
}

X['pre_year'] = pre_year[['discovery_year']].to_numpy(dtype=int).reshape(-1, 1)

# polynomial transformed
X_['pre_year'] = transformer.fit_transform(X['pre_year'])
y['pre_year'] = pre_year[['count_endemic']].to_numpy(dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X_['pre_year'], y['pre_year'], test_size = 0.1, random_state = 53,
                                                    shuffle=True)

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

y_pred_all = model.predict(X_['1900_2021'])
pred_output = np.hstack((X['1900_2021'], y_pred_all, post_1900['count_endemic'].to_numpy(dtype=int).reshape(-1,1)))
loss = pred_output[:,2, None] - pred_output[:,1, None]
pred_output = np.hstack((pred_output, loss))
true_endemic_sum = pred_output[pred_output[:, 0] >= 2000, 2].sum()
loss_sum = pred_output[pred_output[:, 0] >= 2000, 3].sum()

pred_output = pd.DataFrame(pred_output, columns=['discovery_year', 'true_endemic', 'observed_endemic', 'diff'])