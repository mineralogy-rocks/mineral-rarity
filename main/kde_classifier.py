import pandas as pd
import numpy as np

from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

from modules.gsheet_api import GsheetApi
from functions.helpers import Scaler, transform_data


# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Functions for classifying mineral locality counts into clusters/categories
using kernel density estimate
"""


GsheetApi = GsheetApi()
Scaler = Scaler()
GsheetApi.run_main()

locs = GsheetApi.locs.copy()

locs['locality_counts'] = pd.to_numeric(locs['locality_counts'])


# Pre-process data and create a list of arrays of [x, y], where x is the locality count and y is mineral count

raw_locality_mineral_pairs, log_locality_mineral_pairs, scaled_locality_1d = transform_data(locs, Scaler)


# Kernel Density Estimate (gaussian, tophat, epanechnikov on 1-d data)

N = len(locs)

X = scaled_locality_1d
x_min, x_max = scaled_locality_1d.min(), scaled_locality_1d.max()

X_plot = np.linspace(x_min - 1, x_max + 1, 1000)[:, np.newaxis]

true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(10, 1).pdf(X_plot[:, 0])

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")
colors = ["navy", "cornflowerblue", "darkorange"]
kernels = ["gaussian", "tophat", "epanechnikov"]
lw = 1.5

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.4).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        color=color,
        lw=lw,
        linestyle="-",
        label="kernel = '{0}'".format(kernel),
    )

ax.text(4, 0.44, "N={0} points".format(N))

ax.legend(loc="upper right")
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "|k")


plt.savefig(f"figures/kde/classification_output.jpeg", dpi=300, format='jpeg')

plt.close()


## Kernel Density Estimate (epanechnikov) on 1-d data with different bandwidths

N = len(locs)

X = scaled_locality_1d
x_min, x_max = scaled_locality_1d.min(), scaled_locality_1d.max()

X_plot = np.linspace(x_min - 1, x_max + 1, 1000)[:, np.newaxis]

true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(10, 1).pdf(X_plot[:, 0])

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")
colors = ['black', 'blue', 'darkorange', 'brown']

for index, bandwidth in enumerate(np.arange(0.2, 0.6, 0.1)):
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        color=colors[index],
        lw=0.8,
        linestyle="--",
        label="KDE with bandwidth = {}".format(round(bandwidth, 2)),
    )

ax.text(4, 0.44, "N={0} points".format(N))

ax.legend(loc="upper right")
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "|k")


plt.savefig(f"figures/kde/classification_output_n_bandwidth.jpeg", dpi=300, format='jpeg')

plt.close()


## Get decision boundaries

X = scaled_locality_1d
X.sort()
x_min, x_max = scaled_locality_1d.min(), scaled_locality_1d.max()

X_plot = np.linspace(x_min - 1, x_max + 1, 1000)[:, np.newaxis]

kde = KernelDensity(kernel="epanechnikov", bandwidth=0.3)
kde.fit(X)

log_dens = kde.score_samples(X_plot)

mi, ma = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]
clusters = np.concatenate([scaled_locality_1d[mi], scaled_locality_1d[ma]])
clusters = np.unique(clusters)


# Obtain labels for each point in test array. Use last trained model.

log_descaled = Scaler.descale_features(clusters.reshape(-1,1))

descaled = np.exp(log_descaled)
descaled = np.sort(descaled, axis=0)
descaled = descaled.ravel()

print(descaled)