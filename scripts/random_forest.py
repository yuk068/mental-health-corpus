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

n_components = 100

lsa = TruncatedSVD(n_components=n_components, random_state=42)
X_train_tfidf = lsa.fit_transform(X_train_tfidf)
X_test_tfidf = lsa.transform(X_test_tfidf)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model on the training data
rf_model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report for detailed metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix for additional insights
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))