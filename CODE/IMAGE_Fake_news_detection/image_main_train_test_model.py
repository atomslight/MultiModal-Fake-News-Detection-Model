#Cross validation modified code of training and testing of image
import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from scipy.fftpack import dct
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib

# Function to extract features from an image
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG Features
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Color Features (Mean, Standard Deviation, Skewness)
    color_features = []
    for channel in cv2.split(image):
        color_features.extend([np.mean(channel), np.std(channel), skew(channel.flatten())])
    color_features = np.array(color_features)

    # LBP Features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 59), density=True)

    # DCT Features
    dct_features = dct(dct(gray, axis=0), axis=1).flatten()

    # Color Histogram Features
    color_hist_features = []
    for channel in cv2.split(image):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        color_hist_features.extend(hist.flatten())
    color_hist_features = np.array(color_hist_features)

    # Edge Detection Features (Canny Edge Detector)
    edges = cv2.Canny(gray, 100, 200)
    edge_hist, _ = np.histogram(edges, bins=np.arange(0, 256), density=True)

    # Combine All Features
    features = np.concatenate([hog_features, color_features, lbp_hist.flatten(), dct_features, color_hist_features, edge_hist.flatten()])
    return features

# Load images and labels
def load_images_from_folder(folder, label, image_list, label_list, img_size):
    files = os.listdir(folder)
    for filename in files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize the image here
            image_list.append(img)
            label_list.append(label)

# Directories for real and fake images
real_images_dir = '/content/drive/MyDrive/training twitter/gossipcop_real'
fake_images_dir = '/content/drive/MyDrive/training twitter/gossipcop_fake'

# Lists to store images and labels
images = []
labels = []

# Desired image size
image_size = (256, 256)

# Load all real and fake images
load_images_from_folder(real_images_dir, 1, images, labels, image_size)
load_images_from_folder(fake_images_dir, 0, images, labels, image_size)

# Extract features for all images
features = np.array([extract_features(image) for image in images])

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert labels to numpy array
labels = np.array(labels)

# Split the dataset using StratifiedKFold for better generalization
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []

for train_index, test_index in kfold.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Random Forest model and parameter grid
    rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    # GridSearchCV for Random Forest
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    best_rf = grid_search_rf.best_estimator_

    # Predict on test set with Random Forest
    y_test_pred_rf = best_rf.predict(X_test)

    # Evaluate the model on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred_rf)
    precision = precision_score(y_test, y_test_pred_rf)
    recall = recall_score(y_test, y_test_pred_rf)
    f1 = f1_score(y_test, y_test_pred_rf)

    # Append metrics to lists
    test_accuracies.append(test_accuracy)
    test_precisions.append(precision)
    test_recalls.append(recall)
    test_f1_scores.append(f1)

# Calculate average metrics across all folds
avg_test_accuracy = np.mean(test_accuracies)
avg_test_precision = np.mean(test_precisions)
avg_test_recall = np.mean(test_recalls)
avg_test_f1 = np.mean(test_f1_scores)

# Save average metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [avg_test_accuracy, avg_test_precision, avg_test_recall, avg_test_f1]
})
metrics_df.to_csv('metrics_images.csv', index=False)

# Save the trained model to a file
joblib.dump(best_rf, 'random_forest_model.pkl')

# Load images for mixed image prediction
def load_images_for_testing(folder, img_size):
    images = []
    filenames = []
    files = os.listdir(folder)
    for filename in files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Directory for mixed images
mixed_images_dir = '/content/drive/MyDrive/mix_images_twitter'

# Load mixed images
test_images, test_filenames = load_images_for_testing(mixed_images_dir, image_size)

# Extract features for mixed images
test_features_mixed = np.array([extract_features(image) for image in test_images])

# Normalize test features
test_features_mixed = scaler.transform(test_features_mixed)

# Load the trained model from a file
best_rf = joblib.load('random_forest_model.pkl')

# Predict on mixed images with Random Forest
test_predictions_mixed = best_rf.predict(test_features_mixed)

# Map numeric predictions to labels
label_mapping = {1: 'real', 0: 'fake'}
predicted_labels = [label_mapping[pred] for pred in test_predictions_mixed]

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Filename': test_filenames,
    'Prediction': predicted_labels,
    'Numeric_Label': test_predictions_mixed
})

# Save predictions to a CSV file
results_df.to_csv('images_predictions.csv', index=False)

# Output the results
print("Predictions for mixed images:")
print(results_df)

print("\nMetrics:")
print(metrics_df)
