import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load the data
cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_corpus['text'],
    cleaned_corpus['label'],
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# LSA Transformation
lsa = TruncatedSVD(n_components=100, random_state=42)
X_train_lsa = lsa.fit_transform(X_train_tfidf)
X_test_lsa = lsa.transform(X_test_tfidf)

import numpy as np
import random


def analyze_sparsity(tfidf_matrix, lsa_matrix, n_samples=10):
    if not isinstance(tfidf_matrix, np.ndarray):
        tfidf_array = tfidf_matrix.toarray()
    else:
        tfidf_array = tfidf_matrix

    # Get total dimensions
    n_docs = tfidf_array.shape[0]

    # Randomly sample document indices
    sample_indices = random.sample(range(n_docs), n_samples)

    results = []
    for idx in sample_indices:
        # Analyze TF-IDF document
        tfidf_doc = tfidf_array[idx]
        tfidf_nonzero = np.count_nonzero(tfidf_doc)
        tfidf_zero = len(tfidf_doc) - tfidf_nonzero

        # Analyze LSA document
        lsa_doc = lsa_matrix[idx]
        lsa_nonzero = np.count_nonzero(lsa_doc)
        lsa_zero = len(lsa_doc) - lsa_nonzero

        results.append((idx, tfidf_nonzero, tfidf_zero, lsa_nonzero, lsa_zero))

    return results


# Analyze training set
results = analyze_sparsity(X_train_tfidf, X_train_lsa)

# Print results
print("Analysis of 10 random documents:")
print("\nFormat: (Document Index, TF-IDF Non-zero, TF-IDF Zero, LSA Non-zero, LSA Zero)")
print("-" * 75)
for idx, tfidf_nz, tfidf_z, lsa_nz, lsa_z in results:
    print(f"Doc {idx:4d}: {tfidf_nz:4d} non-zero, {tfidf_z:4d} zero  →  {lsa_nz:3d} non-zero, {lsa_z:3d} zero")

# Calculate and print averages
avg_tfidf_nz = np.mean([r[1] for r in results])
avg_lsa_nz = np.mean([r[3] for r in results])
print("\nAverages:")
print(f"TF-IDF: {avg_tfidf_nz:.1f} non-zero elements per document")
print(f"LSA:    {avg_lsa_nz:.1f} non-zero elements per document")