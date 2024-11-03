import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

cleaned_corpus = pd.read_csv('data/cleaned_mhc.csv')

X_train, X_test, y_train, y_test = train_test_split(cleaned_corpus['text'], cleaned_corpus['label'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=3500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=10, random_state=122)
rf_classifier.fit(X_train_tfidf, y_train)

y_pred = rf_classifier.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))

individual_tree = rf_classifier.estimators_[0]

plt.figure(figsize=(10, 8))
plot_tree(
    individual_tree,
    feature_names=tfidf.get_feature_names_out(),
    class_names=['0', '1'],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3
)

plt.savefig("depth_3.png", dpi=300, bbox_inches='tight')
plt.show()