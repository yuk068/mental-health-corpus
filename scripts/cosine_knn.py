import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
cleaned_corpus = pd.read_csv('data/mental_health.csv')

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

# Values of k to test
k_values = [1, 5, 10, 50, 100, 200, 500, 2000, 5000]

# Loop through each k and evaluate the model
for k in k_values:
    print(f"\nEvaluating KNN with k={k}")

    # Initialize KNN with cosine distance
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')

    # Fit the model
    knn.fit(X_train_tfidf, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test_tfidf)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
