import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from modules.rruff_api import RruffApi

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Functions for classifying mineral locality counts into clusters/categories
"""

RruffApi = RruffApi()
RruffApi.run_main()


locs = RruffApi.locs


# Pre-process data and create a list of arrays of [x, y], where x is the locality count and y is mineral count

locs_2016 = locs.loc[locs['locality_counts_2016'].notna()][['locality_counts_2016']]
locs_2016['mineral_count'] = 1

locs_2016 = locs_2016.groupby(by='locality_counts_2016').agg(mineral_count=pd.NamedAgg(column="mineral_count", aggfunc="sum")).sort_values(by='mineral_count', ascending=False).reset_index(level=0)

locs_2016['locality_counts_2016'] = pd.to_numeric(locs_2016['locality_counts_2016'])
locs_2016['mineral_count'] = pd.to_numeric(locs_2016['mineral_count'])

locs_2016 = locs_2016.to_numpy()


# Transform features by scaling each feature to a (0, 1)

mms = MinMaxScaler()
mms.fit(locs_2016)
locs_2016_transformed = mms.transform(locs_2016)


# Find optimum number of clusters
## Elbow method

sum_of_squared_distances = []
range_n_clusters = range(1, 15)

for n_clusters in range_n_clusters:
    km = KMeans(n_clusters=n_clusters)
    km = km.fit(locs_2016_transformed)
    sum_of_squared_distances.append(km.inertia_)


plt.plot(range_n_clusters, sum_of_squared_distances, color='green', marker='o',
         linestyle='solid', linewidth=1, markersize=2)

plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal Number of Clusters')

plt.savefig("figures/elbow_n_clusters.jpeg", dpi=300, format='jpeg')

plt.close()

## silhouette method




kmeans = KMeans(n_clusters=6).fit(locs_2016)

kmeans.cluster_centers_

kmeans.n_features_in_

kmeans = KMeans(n_clusters=6).fit_predict(locs_2016)

kmeans.cluster_centers_

kmeans.n_features_in_