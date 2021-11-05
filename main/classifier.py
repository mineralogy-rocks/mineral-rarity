import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KernelDensity

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from modules.gsheet_api import GsheetApi
from functions.helpers import Scaler, transform_data

# -*- coding: utf-8 -*-
"""
@author: Liubomyr Gavryliv
Functions for classifying mineral locality counts into clusters/categories
using k-means clustering and kernel density estimate
"""

GsheetApi = GsheetApi()
Scaler = Scaler()
GsheetApi.run_main()

locs = GsheetApi.locs.copy()


# Pre-process data and create a list of arrays of [x, y], where x is the locality count and y is mineral count

raw_locality_mineral_pairs, log_locality_mineral_pairs, scaled_locality_1d = transform_data(locs, Scaler)


# Original non-scaled data scatter-plot

plt.scatter(raw_locality_mineral_pairs['locality_counts'], raw_locality_mineral_pairs['mineral_count'], color='#5FD6D1', marker='o', s=20,
            edgecolors='black', linewidths=0.1)

plt.xlabel('Locality count')
plt.ylabel('Mineral count')
plt.title('Mineral - Locality pairs')

plt.savefig(f"figures/mineral_locality_pairs.jpeg", dpi=300, format='jpeg')

plt.close()


# Log-transformed data scatter-plot

plt.scatter(log_locality_mineral_pairs['locality_counts'], log_locality_mineral_pairs['mineral_count'], color='#5FD6D1', marker='o', s=20,
            edgecolors='black', linewidths=0.1)

plt.xlabel('Log (Locality count)')
plt.ylabel('Log (Mineral count)')
plt.title('Mineral - Locality pairs')

plt.savefig(f"figures/log_mineral_locality_pairs.jpeg", dpi=300, format='jpeg')

plt.close()


# Original non-scaled locality counts histogram

plt.hist(raw_locality_mineral_pairs['locality_counts'], bins=1000)

plt.xlabel('Locality count')
plt.ylabel('Mineral count')
plt.title('Mineral - Locality pairs')

plt.savefig(f"figures/raw_locality_counts_histogram.jpeg", dpi=300, format='jpeg')

plt.close()


# Standardized and Scaled locality counts histogram

plt.hist(scaled_locality_1d, bins=1000)

plt.xlabel('Locality count')
plt.ylabel('Mineral count')
plt.title('Mineral - Locality pairs')

plt.savefig(f"figures/scaled_locality_counts_histogram.jpeg", dpi=300, format='jpeg')

plt.close()

# Find optimum number of clusters for k-Means

N_CLUSTERS = range(2, 15)

## Elbow method

sum_of_squared_distances = []

for n_clusters in N_CLUSTERS:
    km = KMeans(n_clusters=n_clusters)
    km = km.fit(scaled_locality_1d)
    sum_of_squared_distances.append(km.inertia_)


plt.plot(N_CLUSTERS, sum_of_squared_distances, color='green', marker='o',
         linestyle='solid', linewidth=1, markersize=2)

plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal Number of Clusters')

plt.savefig("figures/k-means/elbow_n_clusters.jpeg", dpi=300, format='jpeg')

plt.close('all')


## Silhouette method

for n_clusters in range(2,15):

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this classification all
    # lie within [-0.1, 1]

    ax1.set_xlim([-0.1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.

    ax1.set_ylim([0, len(scaled_locality_1d) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(scaled_locality_1d)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    silhouette_avg = silhouette_score(scaled_locality_1d, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(scaled_locality_1d, cluster_labels)

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
        scaled_locality_1d[:, 0], scaled_locality_1d[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
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

    plt.savefig(f"figures/k-means/silhouette_{n_clusters}_clusters.jpeg", dpi=300, format='jpeg')

    plt.close('all')


## K-means classification 1-D (frequencies of normalized and scaled locality counts)

N_CLUSTERS = 6

kmeans = KMeans(n_clusters=N_CLUSTERS)

kmeans.fit(scaled_locality_1d)

pred_classes = kmeans.predict(scaled_locality_1d)

for n_cluster in range(N_CLUSTERS):
    locs.loc[locs.index.isin(*np.where(pred_classes == n_cluster)), 'cluster'] = n_cluster


plt.scatter(locs['locality_counts'].to_numpy(dtype=int).reshape(-1,1)[:, 0], locs.index.to_numpy(), c=pred_classes, s=50, cmap='viridis')

plt.savefig(f"figures/k-means/classification_output.jpeg", dpi=300, format='jpeg')

plt.close()


### Get decision boundaries for each cluster by running a prediction

# Step size of the test set. Decrease to increase the quality of the VQ.
h = 0.02

# These are already scaled to locs_min=1 and locs_max set above, cause the source of data remains the same
x_min, x_max = scaled_locality_1d.min(), scaled_locality_1d.max()

xx = np.arange(x_min, x_max, h)

# Obtain labels for each point in test array. Use last trained model.

xx_predicated = kmeans.predict(xx.reshape(-1,1))

log_descaled = Scaler.descale_features(xx.reshape(-1,1))

descaled = np.exp(log_descaled)
descaled = np.sort(descaled, axis=0)
descaled = descaled.ravel()

locs_classes = pd.DataFrame({ 'locality_counts': descaled, 'predicted': xx_predicated })

locs_classes = locs_classes.groupby('predicted')[['locality_counts']].agg(lambda x: str(np.floor(x.min())) + '-' + str(np.floor(x.max())))

locs_classes.sort_values('locality_counts', inplace=True)


### Classification according to k-means 1-d

# Rare minerals
# 1
# 2-4
# 5-16

# Transitional minerals
# 17-75

# Widespread minerals
# 76-590
# > 590


# Kernel density estimation (frequencies of raw locality counts)

kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(scaled_locality_1d)

s = np.linspace(0,80000)
e = kde.score_samples(s.reshape(-1,1))
plt.plot(s, e)

plt.savefig(f"figures/k-means/classification_output.jpeg", dpi=300, format='jpeg')

plt.close()