import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

cosine_sim_matrix = cosine_similarity(tfidf_matrix)
n_components = 3
lsa = TruncatedSVD(n_components=n_components, random_state=42)
doc_coords = lsa.fit_transform(cosine_sim_matrix)
term_coords = lsa.components_.T * np.sqrt(lsa.singular_values_)

def create_3d_biplot_with_labels(doc_coords, term_coords, terms, labels, n_terms=500, highlight_terms=[]):
    term_distances = np.sqrt(np.sum(term_coords ** 2, axis=1))
    top_term_indices = [i for i in term_distances.argsort()[-n_terms:][::-1] if i < len(terms)]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=term_coords[top_term_indices, 0],
            y=term_coords[top_term_indices, 1],
            z=term_coords[top_term_indices, 2],
            mode='markers+text',
            name='Terms',
            text=[terms[i] for i in top_term_indices],
            textposition="top center",
            marker=dict(
                size=3,
                color=['blue' if terms[i] in highlight_terms else 'red' for i in top_term_indices],
                opacity=0.8,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            textfont=dict(
                size=10,
                color=['blue' if terms[i] in highlight_terms else 'red' for i in top_term_indices]
            ),
        )
    )

    unique_labels = sorted(labels.unique(), reverse=True)
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    color_map = dict(zip(unique_labels, colors))

    for label in unique_labels:
        mask = labels == label
        indices = cleaned_corpus[mask].index
        fig.add_trace(
            go.Scatter3d(
                x=doc_coords[mask, 0],
                y=doc_coords[mask, 1],
                z=doc_coords[mask, 2],
                mode='markers',
                name=f'Documents ({label})',
                marker=dict(
                    size=3,
                    opacity=0.6,
                    color=color_map[label],
                ),
                hovertemplate="Document Index: %{customdata}<br>",
                customdata=indices
            )
        )

    fig.update_layout(
        title='3D LSA Biplot: Documents and Terms',
        scene=dict(
            xaxis_title=f'Component 1',
            yaxis_title=f'Component 2',
            zaxis_title=f'Component 3',
            xaxis=dict(gridcolor='rgb(255, 255, 255)'),
            yaxis=dict(gridcolor='rgb(255, 255, 255)'),
            zaxis=dict(gridcolor='rgb(255, 255, 255)')
        ),
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            title='Categories'
        ),
        width=1600,
        height=1000
    )

    return fig

highlight_terms = ['suicidal', 'depression', 'suicide', 'redflag', 'kill', 'myself', 'die',
                   'died', 'pain', 'sad', 'help', 'please', 'sorry', 'anxiety', 'mental', 'therapy',
                   'therapist']
fig = create_3d_biplot_with_labels(
    doc_coords,
    term_coords,
    tfidf.get_feature_names_out(),
    cleaned_corpus['label'],
    highlight_terms=highlight_terms
)
fig.show()

print("\nLabel Distribution:")
print(cleaned_corpus['label'].value_counts())

def calculate_label_centroids(doc_coords, labels):
    centroids = {}
    for label in labels.unique():
        mask = labels == label
        centroid = doc_coords[mask].mean(axis=0)
        centroids[label] = centroid

    return pd.DataFrame.from_dict(centroids, orient='index',
                                  columns=['Component_1', 'Component_2', 'Component_3'])

centroids_df = calculate_label_centroids(doc_coords, cleaned_corpus['label'])
print("\nLabel Centroids:")
print(centroids_df)
