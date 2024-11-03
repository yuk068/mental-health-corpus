import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the cleaned corpus
cleaned_corpus = pd.read_csv('data/mental_health.csv')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3500)
tfidf_matrix = tfidf.fit_transform(cleaned_corpus['text'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Add the labels to the DataFrame
tfidf_df['label'] = cleaned_corpus['label']

# Calculate mean and standard deviation for each feature based on class
mean_std_class_0 = tfidf_df[tfidf_df['label'] == 0].drop(columns='label').mean().to_frame(name='Mean Class 0')
mean_std_class_0['Standard Deviation Class 0'] = tfidf_df[tfidf_df['label'] == 0].drop(columns='label').std()

mean_std_class_1 = tfidf_df[tfidf_df['label'] == 1].drop(columns='label').mean().to_frame(name='Mean Class 1')
mean_std_class_1['Standard Deviation Class 1'] = tfidf_df[tfidf_df['label'] == 1].drop(columns='label').std()

# Combine the results
mean_std_combined = pd.concat([mean_std_class_0, mean_std_class_1], axis=1)

# Display the combined results
print(mean_std_combined)

# Optionally, save the results to a CSV file
mean_std_combined.to_csv('data/tfidf_mean_std.csv')
