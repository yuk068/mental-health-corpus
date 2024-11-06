import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Load and prepare data
cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

# Test different numbers of components
n_components_range = [5, 15, 50, 100, 150, 200, 300, 500]
explained_variances = []
cumulative_variances = []

for n in n_components_range:
    lsa = TruncatedSVD(n_components=n, random_state=42)
    lsa.fit(tfidf_matrix)
    explained_variances.append(lsa.explained_variance_ratio_)
    cumulative_variances.append(np.sum(lsa.explained_variance_ratio_))

# Plot the results
plt.figure(figsize=(8, 6))

plt.plot(n_components_range, cumulative_variances, 'r-o', linewidth=2)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.grid(True)

plt.tight_layout()

plt.show()

# Print detailed statistics
print("Explained Variance Analysis:")
print("-" * 50)
for n, var in zip(n_components_range, cumulative_variances):
    print(f"Number of components: {n:3d}, Cumulative explained variance: {var:.3f} ({var*100:.1f}%)")

# Find optimal number of components for different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8]
print("\nComponents needed for different variance thresholds:")
print("-" * 50)

for threshold in thresholds:
    for n, var in zip(n_components_range, cumulative_variances):
        if var >= threshold:
            print(f"To explain {threshold*100}% of variance, need {n} components")
            break

# Calculate the "elbow" using rate of change
variance_diffs = np.diff([0] + cumulative_variances)
variance_ratios = variance_diffs / np.array(n_components_range)
elbow_idx = np.argmax(np.diff(variance_ratios) < 0.0001) if len(variance_ratios) > 1 else 0
optimal_components = n_components_range[elbow_idx]

print(f"\nOptimal number of components (elbow method): {optimal_components}")