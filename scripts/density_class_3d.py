import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

# Load and process data
cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

# Perform LSA with 3 components for 3D visualization
lsa = TruncatedSVD(n_components=3, random_state=42)
doc_coords = lsa.fit_transform(tfidf_matrix)

# Use 'label' column as clusters
labels = cleaned_corpus['label']


def create_3d_density_plot(doc_coords, labels):
    # Create 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separate points by label
    mask0 = labels == 0
    mask1 = labels == 1

    label0_points = doc_coords[mask0]
    label1_points = doc_coords[mask1]

    # Plot both labels with the specified pastel colors
    ax.scatter(
        label0_points[:, 0],
        label0_points[:, 1],
        label0_points[:, 2],
        c='#FFFF99',  # Light yellow
        s=50,
        alpha=0.6,
        label='Class 0'
    )

    ax.scatter(
        label1_points[:, 0],
        label1_points[:, 1],
        label1_points[:, 2],
        c='#99FFE5',  # Mint/cyan
        s=50,
        alpha=0.6,
        label='Class 1'
    )

    # Add labels and title
    ax.set_xlabel('LSA Component 1')
    ax.set_ylabel('LSA Component 2')
    ax.set_zlabel('LSA Component 3')
    ax.set_title('3D LSA Components Distribution')

    # Add legend
    ax.legend()

    # Set background color to white for better contrast
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    return plt


# Create and display the 3D plot
plt.style.use('default')
plot = create_3d_density_plot(doc_coords, labels)

# Print label distribution
print("\nLabel Distribution:")
print(cleaned_corpus['label'].value_counts())

plt.show()