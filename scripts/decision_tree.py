import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

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

# Define parameter grid
param_grid = {
    'random_state': [42, 123, 256, 789, 101, 303, 404, 505, 606, 707],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Create base classifier
dt_classifier = DecisionTreeClassifier()

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=dt_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Perform grid search
grid_search.fit(X_train_tfidf, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Make predictions with best model
y_pred = best_model.predict(X_test_tfidf)

# Print results
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest F1 Score from Grid Search:")
print(grid_search.best_score_)

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTree Structure of Best Model:")
print(f"Total number of nodes: {best_model.tree_.node_count}")
print(f"Number of leaves: {best_model.get_n_leaves()}")
print(f"Maximum depth: {best_model.get_depth()}")