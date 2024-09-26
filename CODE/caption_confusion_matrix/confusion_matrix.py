import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Read the CSV files
true_labels = pd.read_csv('/content/true_labels_caption.csv')
predictions = pd.read_csv('/content/caption_predictions.csv')

# Merge the data on 'Filename' to include all entries
merged_df = pd.merge(predictions, true_labels, on='Filename', how='outer')

# Handle missing values by filling NaN with a default value (e.g., -1)
merged_df['Label'].fillna(-1, inplace=True)
merged_df['Numeric_Label'].fillna(-1, inplace=True)

# Extract the true labels and predictions
y_true = merged_df['Label'].astype(int)
y_pred = merged_df['Numeric_Label'].astype(int)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, -1])

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True', 'Missing'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
