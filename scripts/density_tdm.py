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
def plot_lsa_biplot(doc_coords, term_coords, labels, label_order, ax):
    # Calculate point density
    xy = np.vstack([doc_coords[:, 0], doc_coords[:, 1]])
    density = gaussian_kde(xy)(xy)
    density = (density - density.min()) / (density.max() - density.min())
    color_maps = ['summer', 'spring'] if label_order == [1, 0] else ['spring', 'summer']

    # Plot each class in specified order
    for label, color_map in zip(label_order, color_maps):
        mask = labels == label
        ax.scatter(
            doc_coords[mask, 0],
            doc_coords[mask, 1],
            c=density[mask],
            cmap=color_map,
            s=50,
            alpha=0.6
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

    ax.set_title(f'Class {label_order[1]} on Top of Class {label_order[0]}')
    ax.set_xlabel('LSA Component 1')
    ax.set_ylabel('LSA Component 2')
    ax.grid(True, alpha=0.3)

# Create side-by-side subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plot Class 1 on top of Class 0 on the left
plot_lsa_biplot(doc_coords, term_coords, cleaned_corpus['label'], [1, 0], axs[0])

# Plot Class 0 on top of Class 1 on the right
plot_lsa_biplot(doc_coords, term_coords, cleaned_corpus['label'], [0, 1], axs[1])

# Display plots
plt.tight_layout()
# plt.savefig("term_document_density.png", dpi=200, bbox_inches='tight')
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
