import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

curr_locs = locs_2016.to_numpy()


# Transform features by scaling each feature to a (0, 1)

mms = MinMaxScaler()
mms.fit(curr_locs)
curr_locs_transformed = mms.transform(curr_locs)


# Find optimum number of clusters

range_n_clusters = range(2, 15)

## Elbow method

sum_of_squared_distances = []

for n_clusters in range_n_clusters:
    km = KMeans(n_clusters=n_clusters)
    km = km.fit(curr_locs_transformed)
    sum_of_squared_distances.append(km.inertia_)


plt.plot(range_n_clusters, sum_of_squared_distances, color='green', marker='o',
         linestyle='solid', linewidth=1, markersize=2)

plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal Number of Clusters')

plt.savefig("figures/knn/elbow_n_clusters.jpeg", dpi=300, format='jpeg')

plt.close('all')


## Silhouette method

for n_clusters in range_n_clusters:

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this classification all
    # lie within [-0.1, 1]

    ax1.set_xlim([-0.1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.

    ax1.set_ylim([0, len(curr_locs_transformed) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(curr_locs_transformed)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    silhouette_avg = silhouette_score(curr_locs_transformed, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(curr_locs_transformed, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them

        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        curr_locs_transformed[:, 0], curr_locs_transformed[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(f"figures/knn/silhouette_{n_clusters}_clusters.jpeg", dpi=300, format='jpeg')

    plt.close('all')



kmeans = KMeans(n_clusters=6).fit(curr_locs)

kmeans.cluster_centers_

kmeans.n_features_in_

kmeans = KMeans(n_clusters=6).fit_predict(curr_locs)

kmeans.cluster_centers_

kmeans.n_features_in_