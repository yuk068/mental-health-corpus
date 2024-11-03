import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load and process data
cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

# Perform LSA
lsa = TruncatedSVD(n_components=2, random_state=42)
doc_coords = lsa.fit_transform(tfidf_matrix)

# Use 'label' column as clusters
labels = cleaned_corpus['label']


def create_density_plots(doc_coords, labels):
    # Create figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    colors = ['viridis', 'plasma']  # Two consistent color maps

    # Plot for label 0 on top of label 1 in the first subplot
    mask0 = labels == 1
    mask1 = labels == 0

    label0_points = doc_coords[mask0]
    label1_points = doc_coords[mask1]

    # Calculate density for label 0
    xy0 = np.vstack([label0_points[:, 0], label0_points[:, 1]])
    density0 = gaussian_kde(xy0)(xy0)
    density0 = (density0 - density0.min()) / (density0.max() - density0.min())

    # Calculate density for label 1
    xy1 = np.vstack([label1_points[:, 0], label1_points[:, 1]])
    density1 = gaussian_kde(xy1)(xy1)
    density1 = (density1 - density1.min()) / (density1.max() - density1.min())

    # Plot label 0
    axs[0].scatter(
        label0_points[:, 0],
        label0_points[:, 1],
        c=density0,
        cmap=colors[0],
        s=50,
        alpha=0.6
    )
    # Plot label 1 on top
    axs[0].scatter(
        label1_points[:, 0],
        label1_points[:, 1],
        c=density1,
        cmap=colors[1],
        s=50,
        alpha=0.6
    )
    axs[0].set_title('Label 0 on top of Label 1')
    axs[0].grid(True, alpha=0.3)

    # Plot for label 1 on top of label 0 in the second subplot
    axs[1].scatter(
        label1_points[:, 0],
        label1_points[:, 1],
        c=density1,
        cmap=colors[1],
        s=50,
        alpha=0.6
    )
    # Plot label 0 on top
    axs[1].scatter(
        label0_points[:, 0],
        label0_points[:, 1],
        c=density0,
        cmap=colors[0],
        s=50,
        alpha=0.6
    )
    axs[1].set_title('Label 1 on top of Label 0')
    axs[1].grid(True, alpha=0.3)

    # Set common labels for both subplots
    for ax in axs:
        ax.set_xlabel('LSA Component 1')
        ax.set_ylabel('LSA Component 2')

    plt.tight_layout()
    return plt


# Create and display the plots
plt.style.use('default')
plot = create_density_plots(doc_coords, labels)

# Print label distribution
print("\nLabel Distribution:")
print(cleaned_corpus['label'].value_counts())

plt.show()
