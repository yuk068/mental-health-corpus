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
n_components = 2
lsa = TruncatedSVD(n_components=n_components, random_state=42)
doc_coords = lsa.fit_transform(tfidf_matrix)

# Calculate term coordinates
term_coords = lsa.components_.T * np.sqrt(lsa.singular_values_)

# Define highlighted terms
highlight_terms = ['suicidal', 'depression', 'suicide', 'kill', 'myself', 'die',
                   'died', 'pain', 'sad', 'help', 'sorry', 'anxiety', 'therapy',
                   'suffering', 'killing', 'pill',
                   'movie', 'film', 'character', 'story', 'actor', 'performance', 'show',
                   'plot', 'acting']

# Filter for highlighted terms only
terms = tfidf.get_feature_names_out()
highlight_indices = [i for i, term in enumerate(terms) if term in highlight_terms]
highlighted_term_coords = term_coords[highlight_indices]
highlighted_terms = [terms[i] for i in highlight_indices]

# Plotting function
def plot_lsa_biplot(doc_coords, term_coords, labels, ax):
    # Calculate point density
    xy = np.vstack([doc_coords[:, 0], doc_coords[:, 1]])
    density = gaussian_kde(xy)(xy)
    density = (density - density.min()) / (density.max() - density.min())

    # Separate Class 0 and Class 1
    class_0_mask = labels == 0
    class_1_mask = labels == 1

    # Plot Class 0 with summer colormap
    scatter_0 = ax.scatter(
        doc_coords[class_0_mask, 0],
        doc_coords[class_0_mask, 1],
        c=density[class_0_mask],
        cmap='summer',
        s=50,
        alpha=0.6,
        label='Class 0 (green)'
    )

    # Plot Class 1 with spring colormap
    scatter_1 = ax.scatter(
        doc_coords[class_1_mask, 0],
        doc_coords[class_1_mask, 1],
        c=density[class_1_mask],
        cmap='spring',
        s=50,
        alpha=0.6,
        label='Class 1 (pink)'
    )

    # Plot highlighted terms only
    for i, term in enumerate(highlighted_terms):
        ax.text(
            highlighted_term_coords[i, 0],
            highlighted_term_coords[i, 1],
            term,
            color='black',
            alpha=0.7,
            fontsize=8
        )
        ax.scatter(
            highlighted_term_coords[i, 0],
            highlighted_term_coords[i, 1],
            color='black',
            alpha=0.3,
            s=10
        )

    # Create an explicit legend
    ax.legend(loc='upper right')

    ax.set_title('Biplot of Selected Terms with LSA Components')
    ax.set_xlabel('LSA Component 1')
    ax.set_ylabel('LSA Component 2')
    ax.grid(True, alpha=0.3)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot Class 0 on top of Class 1
plot_lsa_biplot(doc_coords, term_coords, cleaned_corpus['label'], ax)

# Display plots
plt.tight_layout()
plt.show()

# Print label distribution
print("\nLabel Distribution:")
print(cleaned_corpus['label'].value_counts())

# Calculate and print label centroids
def calculate_label_centroids(doc_coords, labels):
    centroids = {}
    for label in labels.unique():
        mask = labels == label
        centroid = doc_coords[mask].mean(axis=0)
        centroids[label] = centroid

    return pd.DataFrame.from_dict(centroids, orient='index',
                                  columns=['Component_1', 'Component_2'])

centroids_df = calculate_label_centroids(doc_coords, cleaned_corpus['label'])
print("\nLabel Centroids:")
print(centroids_df)
