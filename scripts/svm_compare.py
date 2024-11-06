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
from sklearn.metrics import classification_report
import numpy as np

# 1. Linear SVM
linear_svm = SVC(kernel='linear', random_state=42)
param_grid_linear = {
    'C': [0.1, 1, 10],
    'class_weight': ['balanced', None]
}

# Grid search with cross-validation
grid_linear = GridSearchCV(
    linear_svm,
    param_grid_linear,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='f1'
)

# Fit and evaluate linear SVM
grid_linear.fit(X_train_tfidf, y_train)
y_pred_linear = grid_linear.predict(X_test_tfidf)

print("Linear SVM Best Parameters:", grid_linear.best_params_)
print("\nLinear SVM Performance:")
print(classification_report(y_test, y_pred_linear))

# 2. RBF Kernel SVM (for comparison)
rbf_svm = SVC(kernel='rbf', random_state=42)
param_grid_rbf = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1],
    'class_weight': ['balanced', None]
}

grid_rbf = GridSearchCV(
    rbf_svm,
    param_grid_rbf,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='f1'
)

# Fit and evaluate RBF SVM
grid_rbf.fit(X_train_tfidf, y_train)
y_pred_rbf = grid_rbf.predict(X_test_tfidf)

print("\nRBF SVM Best Parameters:", grid_rbf.best_params_)
print("\nRBF SVM Performance:")
print(classification_report(y_test, y_pred_rbf))

# Compare training times and memory usage
from time import time
import psutil
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


# Time and memory benchmarking
print("\nPerformance Metrics:")
for model_name, model in [('Linear SVM', grid_linear), ('RBF SVM', grid_rbf)]:
    start_time = time()
    model.fit(X_train_tfidf, y_train)
    train_time = time() - start_time

    memory_usage = get_memory_usage()

    print(f"\n{model_name}:")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")