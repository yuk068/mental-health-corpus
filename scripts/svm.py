import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Initialize GridSearchCV with SVC and the defined parameter grid
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_tfidf, y_train)

# Get the best model from grid search
svm_rbf = grid_search.best_estimator_

# Print best parameters
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Predict on the test set with the best model
y_pred_best = svm_rbf.predict(X_test_tfidf)

# Evaluate the best model
print("Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Function to classify a list of custom texts
def classify_custom_texts(text_list):
    predictions = []
    for text in text_list:
        # Vectorize the custom text
        custom_tfidf = tfidf.transform([text])

        # Predict the class label
        prediction = svm_rbf.predict(custom_tfidf)
        predictions.append((text, prediction[0]))

    return predictions


# Test with multiple custom texts
custom_texts = [
    "please help me i want to end it all",
    "please help me with my homework it just wont end",
    "this character in the westworld movie was having suicidal thoughts",
    "im going to commit suicide"
]

predictions = classify_custom_texts(custom_texts)
for text, label in predictions:
    print(f"Custom Text: '{text}'")
    print(f"Predicted Label: {label}\n")

