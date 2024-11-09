import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
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

n_components = 100

lsa = TruncatedSVD(n_components=n_components, random_state=42)
X_train_tfidf = lsa.fit_transform(X_train_tfidf)
X_test_tfidf = lsa.transform(X_test_tfidf)

# Create and fit the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(18, 18, 8, 8, 8, ),  # You can adjust the size and number of layers
                    activation='relu',  # Activation function
                    solver='adam',  # Optimization solver
                    max_iter=1000,  # Maximum iterations
                    random_state=42,
                    early_stopping=True,  # Stop early if validation score doesn't improve
                    validation_fraction=0.1,  # Fraction of training data to set aside for validation
                    n_iter_no_change=10)  # Number of iterations with no improvement to wait before stopping

mlp.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = mlp.predict(X_test_tfidf)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
