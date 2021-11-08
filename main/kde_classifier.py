import pandas as pd
import numpy as np

from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
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


# Kernel Density Estimate on 1-d data

N = 100

X = locs['locality_counts'].to_numpy(dtype=int)
X.sort()
X = X.reshape(-1, 1)

X = scaled_locality_1d

X_plot = np.linspace(scaled_locality_1d.min() - 2, scaled_locality_1d.max() + 2, 1000)[:, np.newaxis]

true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(10, 1).pdf(X_plot[:, 0])

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")
colors = ["navy", "cornflowerblue", "darkorange"]
kernels = ["gaussian", "tophat", "epanechnikov"]
lw = 2

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

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc="upper right")
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")


plt.savefig(f"figures/kde/classification_output.jpeg", dpi=300, format='jpeg')

plt.close()
