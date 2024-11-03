import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load and process data
cleaned_corpus = pd.read_csv('data/mental_health.csv')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

# Perform K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cleaned_corpus['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Perform LSA
lsa = TruncatedSVD(n_components=2, random_state=42)
doc_coords = lsa.fit_transform(tfidf_matrix)


def create_multi_density_plots(doc_coords, labels):
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    colors = ['viridis', 'magma', 'plasma']

    # Flatten axs for easier iteration
    axs = axs.flatten()

    # Plot with all clusters (first subplot)
    for i in range(n_clusters):
        mask = labels == i
        cluster_points = doc_coords[mask]

        # Calculate density
        xy = np.vstack([cluster_points[:, 0], cluster_points[:, 1]])
        density = gaussian_kde(xy)(xy)
        density = (density - density.min()) / (density.max() - density.min())

        # Plot in first subplot
        axs[0].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=density,
            cmap=colors[i],
            s=50,
            alpha=0.6
        )

    axs[0].set_title('All Clusters')
    axs[0].grid(True, alpha=0.3)

    # Individual cluster plots
    for i in range(n_clusters):
        mask = labels == i
        cluster_points = doc_coords[mask]

        # Calculate density
        xy = np.vstack([cluster_points[:, 0], cluster_points[:, 1]])
        density = gaussian_kde(xy)(xy)
        density = (density - density.min()) / (density.max() - density.min())

        # Plot in respective subplot
        axs[i + 1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=density,
            cmap=colors[i],
            s=50,
            alpha=0.6
        )
        axs[i + 1].set_title(f'Cluster {i}')
        axs[i + 1].grid(True, alpha=0.3)

    # Set common labels for all subplots
    for ax in axs:
        ax.set_xlabel('LSA Component 1')
        ax.set_ylabel('LSA Component 2')

    plt.tight_layout()
    return plt


# Create and display the plots
plt.style.use('default')
plot = create_multi_density_plots(doc_coords, cleaned_corpus['cluster'])

# Print cluster distribution
print("\nCluster Distribution:")
print(cleaned_corpus['cluster'].value_counts())

plt.show()