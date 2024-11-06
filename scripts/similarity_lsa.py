import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

n_components = 100
lsa = TruncatedSVD(n_components=n_components, random_state=42)
doc_coords = lsa.fit_transform(tfidf_matrix)

print(doc_coords.shape)

term_coords = lsa.components_.T * np.sqrt(lsa.singular_values_)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def analyze_similarities(tfidf, term_coords, doc_coords, cleaned_corpus, top_n=10):
    # Calculate term similarities
    term_similarities = cosine_similarity(term_coords)

    # Get feature names
    feature_names = np.array(tfidf.get_feature_names_out())

    # Calculate document similarities
    doc_similarities = cosine_similarity(doc_coords)

    # Function to get top similar items
    def get_top_similar(similarities, items, idx, n=top_n):
        sim_scores = similarities[idx]
        # Get indices of top similar items (excluding self)
        top_indices = np.argsort(sim_scores)[::-1][1:n + 1]

        # Convert items to numpy array if it isn't already
        items_array = np.array(list(items))
        return list(zip(items_array[top_indices], sim_scores[top_indices]))

    # Get some interesting terms to analyze
    interesting_terms = ['suicidal', 'depression', 'suicide', 'kill', 'myself', 'die',
                         'died', 'pain', 'sad', 'help', 'sorry', 'anxiety', 'therapy',
                         'suffering', 'killing', 'pill',
                         'movie', 'film', 'character', 'story', 'actor', 'performance', 'show',
                         'plot', 'acting']
    term_results = {}

    for term in interesting_terms:
        if term in feature_names:
            idx = np.where(feature_names == term)[0][0]
            similar_terms = get_top_similar(term_similarities, feature_names, idx)
            term_results[term] = similar_terms

    # Get some example documents and their similar documents
    sample_docs = [0, 10, 20]  # Example document indices
    doc_results = {}

    # Create array of indices
    doc_indices = np.arange(len(cleaned_corpus))

    for doc_idx in sample_docs:
        similar_docs = get_top_similar(doc_similarities, doc_indices, doc_idx)
        doc_results[doc_idx] = {
            'original': cleaned_corpus['text'].iloc[doc_idx][:200] + "...",  # Show first 200 chars
            'original_label': cleaned_corpus['label'].iloc[doc_idx],
            'similar': [(cleaned_corpus['text'].iloc[int(idx)][:200] + "...",
                         cleaned_corpus['label'].iloc[int(idx)],
                         score)
                        for idx, score in similar_docs]
        }

    return term_results, doc_results


# Run the analysis
term_results, doc_results = analyze_similarities(tfidf, term_coords, doc_coords, cleaned_corpus)

# Display results
print("Similar Terms Analysis:")
print("-" * 50)
for term, similars in term_results.items():
    print(f"\nSimilar terms to '{term}':")
    for similar_term, score in similars:
        print(f"  {similar_term}: {score:.3f}")

print("\nSimilar Documents Analysis:")
print("-" * 50)
for doc_idx, results in doc_results.items():
    print(f"\nOriginal Document {doc_idx} [Class: {results['original_label']}]:")
    print(results['original'])
    print("\nSimilar Documents:")
    for i, (doc, label, score) in enumerate(results['similar'], 1):
        print(f"\n{i}. Similarity Score: {score:.3f} [Class: {label}]")
        print(doc)