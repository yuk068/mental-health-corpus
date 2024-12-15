import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import plotly.graph_objects as go
import plotly.express as px

# Load cleaned corpus
cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

# LSA dimensionality reduction to 100 components
lsa_for_clustering = TruncatedSVD(n_components=100, random_state=42)
lsa_matrix = lsa_for_clustering.fit_transform(tfidf_matrix)

# K-Means clustering on LSA-reduced dataset
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cleaned_corpus['cluster'] = kmeans.fit_predict(lsa_matrix)

# LSA dimensionality reduction to 3 components for visualization
lsa_for_visualization = TruncatedSVD(n_components=3, random_state=42)
doc_coords = lsa_for_visualization.fit_transform(tfidf_matrix)

def create_3d_scatter_plot(doc_coords, labels):
    fig = go.Figure()

    unique_labels = sorted(labels.unique())
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    color_map = dict(zip(unique_labels, colors))

    for label in unique_labels:
        mask = labels == label
        fig.add_trace(
            go.Scatter3d(
                x=doc_coords[mask, 0],
                y=doc_coords[mask, 1],
                z=doc_coords[mask, 2],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(
                    size=3,
                    opacity=0.6,
                    color=color_map[label]
                ),
                hovertemplate="Cluster: " + str(label) + "<br>" +
                              "x: %{x:.2f}<br>" +
                              "y: %{y:.2f}<br>" +
                              "z: %{z:.2f}<br>"
            )
        )

    fig.update_layout(
        title='3D LSA Biplot: K-Means Clusters',
        scene=dict(
            xaxis_title=f'LSA Component 1',
            yaxis_title=f'LSA Component 2',
            zaxis_title=f'LSA Component 3',
            xaxis=dict(gridcolor='rgb(255, 255, 255)'),
            yaxis=dict(gridcolor='rgb(255, 255, 255)'),
            zaxis=dict(gridcolor='rgb(255, 255, 255)')
        ),
        showlegend=True,
        width=1600,
        height=1000
    )

    return fig

# Create and display the 3D scatter plot
fig = create_3d_scatter_plot(doc_coords, cleaned_corpus['cluster'])
fig.show()

# Print cluster distribution
print("\nCluster Distribution:")
print(cleaned_corpus['cluster'].value_counts())
