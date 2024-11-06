import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data
cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])


def analyze_similarities_tfidf(tfidf, tfidf_matrix, cleaned_corpus, top_n=10):
    # Get feature names
    feature_names = np.array(tfidf.get_feature_names_out())

    # Calculate document similarities
    print("Calculating document similarities...")
    doc_similarities = cosine_similarity(tfidf_matrix)

    # Calculate term similarities
    print("Calculating term similarities...")
    # Get dense term matrix (terms × documents)
    term_matrix = tfidf_matrix.T.toarray()
    term_similarities = cosine_similarity(term_matrix)

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

    print("Analyzing term similarities...")
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

    print("Analyzing document similarities...")
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
print("Starting similarity analysis...")
term_results, doc_results = analyze_similarities_tfidf(tfidf, tfidf_matrix, cleaned_corpus)

# Display results
print("\nSimilar Terms Analysis:")
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

# Add some basic statistics about similarity scores
print("\nSimilarity Statistics:")
print("-" * 50)
for doc_idx, results in doc_results.items():
    similar_scores = [score for _, _, score in results['similar']]
    same_class = sum(1 for _, label, _ in results['similar']
                     if label == results['original_label'])

    print(f"\nDocument {doc_idx} [Class: {results['original_label']}]:")
    print(f"Average similarity score: {np.mean(similar_scores):.3f}")
    print(f"Max similarity score: {np.max(similar_scores):.3f}")
    print(f"Min similarity score: {np.min(similar_scores):.3f}")
    print(f"Similar documents from same class: {same_class}/{len(results['similar'])}")