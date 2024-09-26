import pandas as pd
import os

# Function to get base filename without extension
def get_base_filename(filename):
    return os.path.splitext(os.path.splitext(filename)[0])[0]

# Load predictions and metrics from the two CSV files
predictions_file1 = '/content/images_predictions.csv'  # Path to the first predictions file
predictions_file2 = '/content/caption_predictions.csv'  # Path to the second predictions file

# Read the predictions CSV files
df1 = pd.read_csv(predictions_file1)
df2 = pd.read_csv(predictions_file2)

# Add a column with the base filename (without extension)
df1['Base_Filename'] = df1['Filename'].apply(get_base_filename)
df2['Base_Filename'] = df2['Filename'].apply(get_base_filename)

# Find common base filenames between the two files
common_base_filenames = set(df1['Base_Filename']).intersection(set(df2['Base_Filename']))

# Filter rows in both dataframes to include only those with common base filenames
df1_filtered = df1[df1['Base_Filename'].isin(common_base_filenames)]
df2_filtered = df2[df2['Base_Filename'].isin(common_base_filenames)]

# Merge the filtered dataframes on the base filename
merged_df = pd.merge(df1_filtered, df2_filtered, on='Base_Filename', suffixes=('_1', '_2'))

# Compute the average of metrics (Assuming columns 'Metric1' and 'Metric2' exist in both files)
# If your metric files have different names, adjust accordingly.
metrics_file1 = '/content/metrics_images.csv'  # Path to the metrics file for predictions_file1
metrics_file2 = '/content/metrics_caption.csv'  # Path to the metrics file for predictions_file2

# Read the metrics CSV files
metrics_df1 = pd.read_csv(metrics_file1)
metrics_df2 = pd.read_csv(metrics_file2)

# Assuming the metrics files have a single row of values
metrics1 = metrics_df1.set_index('Metric')['Value'].to_dict()
metrics2 = metrics_df2.set_index('Metric')['Score'].to_dict()

# Create metric columns for the merged DataFrame
for metric in metrics1.keys():
    merged_df[f'{metric}_1'] = metrics1[metric]
    merged_df[f'{metric}_2'] = metrics2.get(metric, None)  # If the metric is missing in the second file

# Compute the average of metrics
for metric in metrics1.keys():
    merged_df[f'{metric}_Avg'] = (merged_df[f'{metric}_1'] + merged_df[f'{metric}_2']) / 2

# Create the new combined prediction DataFrame with averaged metrics
combined_predictions_df = merged_df[['Filename_1', 'Prediction_1', 'Prediction_2'] + [f'{metric}_Avg' for metric in metrics1.keys()]].rename(columns={
    'Filename_1': 'Filename',
    'Prediction_1': 'Prediction_File1',
    'Prediction_2': 'Prediction_File2'
})

# Save the combined predictions and averaged metrics to a new CSV file
combined_predictions_df.to_csv('prediction_combined_with_metrics.csv', index=False)

print("Combined predictions with metrics saved to 'prediction_combined_with_metrics.csv'")
