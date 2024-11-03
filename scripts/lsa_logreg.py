import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

# Load the data
cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

# Separate features and target
X = cleaned_corpus['text']
y = cleaned_corpus['label']

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
original_accuracies = []
lsa_accuracies = []

for train_index, test_index in kf.split(X, y):
    # Split data into current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Step 1: TF-IDF Vectorization (fit on training set only)
    tfidf = TfidfVectorizer(max_features=3500)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Step 2: LSA dimensionality reduction on TF-IDF
    # Initial SVD fit on training TF-IDF data to find components for 80% variance
    initial_svd = TruncatedSVD(n_components=1500, random_state=42)
    initial_svd.fit(X_train_tfidf)
    cumulative_variance = np.cumsum(initial_svd.explained_variance_ratio_)
    n_components_80_var = np.searchsorted(cumulative_variance, 0.8) + 1

    # Fit SVD with selected number of components on training set only
    svd = TruncatedSVD(n_components=n_components_80_var, random_state=42)
    X_train_lsa = svd.fit_transform(X_train_tfidf)
    X_test_lsa = svd.transform(X_test_tfidf)

    # Step 3: Train and evaluate Logistic Regression on Original TF-IDF data
    log_reg_original = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_original.fit(X_train_tfidf, y_train)
    y_pred_original = log_reg_original.predict(X_test_tfidf)
    original_accuracy = accuracy_score(y_test, y_pred_original)
    original_accuracies.append(original_accuracy)

    # Step 4: Train and evaluate Logistic Regression on LSA-Reduced data
    log_reg_lsa = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_lsa.fit(X_train_lsa, y_train)
    y_pred_lsa = log_reg_lsa.predict(X_test_lsa)
    lsa_accuracy = accuracy_score(y_test, y_pred_lsa)
    lsa_accuracies.append(lsa_accuracy)

# Print cross-validation results
print("Cross-Validation Results on Original TF-IDF Data:")
print("Mean Accuracy:", np.mean(original_accuracies))
print("Standard Deviation:", np.std(original_accuracies))

print("\nCross-Validation Results on LSA-Reduced Data:")
print("Mean Accuracy:", np.mean(lsa_accuracies))
print("Standard Deviation:", np.std(lsa_accuracies))
