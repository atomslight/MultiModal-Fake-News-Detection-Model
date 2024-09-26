#combined code of both training and test data for caption
import os
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack
import joblib

# Load text data
def load_texts(folder_path, label):
    texts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                filenames.append(filename)
    return texts, filenames

# Prepare datasets
real_texts, real_filenames = load_texts('/content/drive/MyDrive/CAPTION_COMBINED Training data/ndtv_captions', 'real')
fake_texts, fake_filenames = load_texts('/content/drive/MyDrive/CAPTION_COMBINED Training data/fakenewsnetwork_captions', 'fake')

texts = real_texts + fake_texts
filenames = real_filenames + fake_filenames
labels = ['real'] * len(real_texts) + ['fake'] * len(fake_texts)

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

texts = [preprocess_text(text) for text in texts]

# Feature Extraction
# Lexical features
vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Unigrams, bigrams, and trigrams
X_lexical = vectorizer.fit_transform(texts)

# Semantic features
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return [sentiment['pos'], sentiment['neu'], sentiment['neg'], sentiment['compound']]

X_semantic = np.array([get_sentiment(text) for text in texts])

# Statistical features
def get_statistical_features(text):
    punctuation_count = sum([text.count(p) for p in string.punctuation])
    word_count = len(text.split())
    return [punctuation_count, word_count]

X_statistical = np.array([get_statistical_features(text) for text in texts])

# Combine features
X = hstack([X_lexical, X_semantic, X_statistical])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train-Test Split
X_train, X_temp, y_train, y_temp, train_filenames, temp_filenames = train_test_split(X, y, filenames, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test, val_filenames, test_filenames = train_test_split(X_temp, y_temp, temp_filenames, test_size=0.5, stratify=y_temp, random_state=42)

# Save true labels for test data
true_labels_df = pd.DataFrame({
    'Filename': test_filenames,
    'Label': y_test
})
true_labels_df.to_csv('true_labels_caption.csv', index=False)

# Model Training and Evaluation with Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model.fit(X_train_fold, y_train_fold)
    y_val_pred = model.predict(X_val_fold)
    print("Fold Classification Report:\n", classification_report(y_val_fold, y_val_pred))

# Final evaluation
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print("Final Classification Report:\n", classification_report(y_test, y_test_pred))

# Save model and vectorizer
joblib.dump(model, 'rf_model_caption.pkl')
joblib.dump(vectorizer, 'vectorizer_caption.pkl')
joblib.dump(label_encoder, 'label_encoder_caption.pkl')

print("Model and vectorizer saved.")

# Load model, vectorizer, and label encoder
model = joblib.load('rf_model_caption.pkl')
vectorizer = joblib.load('vectorizer_caption.pkl')
label_encoder = joblib.load('label_encoder_caption.pkl')

# Load new text data for testing
def load_texts(folder_path):
    texts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                filenames.append(filename)
    return texts, filenames

# Prepare test data
test_texts, test_filenames = load_texts('/content/drive/MyDrive/CAPTION_COMBINED TEST DATA')

test_texts = [preprocess_text(text) for text in test_texts]

# Feature Extraction
# Lexical features
X_lexical = vectorizer.transform(test_texts)

# Semantic features
X_semantic = np.array([get_sentiment(text) for text in test_texts])

# Statistical features
X_statistical = np.array([get_statistical_features(text) for text in test_texts])

# Combine features
X = hstack([X_lexical, X_semantic, X_statistical])

# Predict on new text data
test_predictions = model.predict(X)

# Map numeric predictions to labels
label_mapping = {1: 'real', 0: 'fake'}
predicted_labels = [label_mapping[pred] for pred in test_predictions]

# Convert 'real' and 'fake' to 1 and 0
numeric_labels = [1 if label == 'real' else 0 for label in predicted_labels]

# Save predictions to CSV
results_df = pd.DataFrame({
    'Filename': test_filenames,
    'Prediction': predicted_labels,
    'Numeric_Label': numeric_labels
})
results_df.to_csv('caption_predictions.csv', index=False)

# Output the results
print("Predictions for test texts:")
print(results_df)

# Compute metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('metrics_caption.csv', index=False)

print("Metrics saved to metrics.csv")
