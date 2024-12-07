import pandas as pd
from sklearn.model_selection import train_test_split
import json
from google.cloud import storage
import logging
import io
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load data from GCS
def load_data_from_gcs(bucket_name, file_name):
    """
    Load CSV data from a Google Cloud Storage bucket.
    """
    logging.info(f"Loading data from GCS: {bucket_name}/{file_name}")
    client = storage.Client(project="recommender-system-gcp-k8s")  # Ensure project is specified
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_text()
    return pd.read_csv(io.StringIO(data)) 

# Save data to GCS
def save_data_to_gcs(bucket_name, file_name, data):
    """
    Save a Pandas DataFrame as a CSV file to a Google Cloud Storage bucket.
    """
    logging.info(f"Saving data to GCS: {bucket_name}/{file_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data.to_csv(index=False), 'text/csv')

# Encode user and product IDs
def encode_data(df):
    """
    Encode user_id and product_id to numerical IDs.
    """
    logging.info("Encoding user_id and product_id")
    user_encoder = {user: idx for idx, user in enumerate(df['user_id'].unique())}
    product_encoder = {product: idx for idx, product in enumerate(df['product_id'].unique())}
    df['user_id'] = df['user_id'].map(user_encoder)
    df['product_id'] = df['product_id'].map(product_encoder)
    return df, user_encoder, product_encoder

# Save encoders to GCS
def save_encoders_to_gcs(bucket_name, encoders, file_name):
    """
    Save user and product encoders as a JSON file to GCS.
    Converts numpy.int64 keys to standard Python integers for JSON compatibility.
    """
    logging.info(f"Saving encoders to GCS: {bucket_name}/{file_name}")
    # Convert numpy.int64 keys to Python int
    encoders = {
        key: {int(k): v for k, v in value.items()}
        for key, value in encoders.items()
    }
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(encoders))


# Main preprocessing function
def preprocess():
    """
    Main function to preprocess data, encode IDs, split into train/test, 
    and save outputs to GCS.
    """
    # GCS bucket and file paths
    bucket_name = "recommender-system-bucket"
    raw_file_name = "cleaned_events.csv"
    train_file_name = "processed/train.csv"
    test_file_name = "processed/test.csv"
    encoders_file_name = "processed/encoders.json"
    
    # Load data
    df = load_data_from_gcs(bucket_name, raw_file_name)
    
    # Convert 'event_time' to a numeric timestamp
    df['interaction_strength'] = pd.to_datetime(df['event_time']).astype(int) / 10**9  # Convert to seconds

    # Filter for 'view' events only
    logging.info("Filtering 'view' events")
    df = df[df['event_type'] == 'view']
    
    # Encode user and product IDs
    df, user_encoder, product_encoder = encode_data(df)
    
    # Create train/test split
    logging.info("Splitting data into train and test sets")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save train/test datasets to GCS
    save_data_to_gcs(bucket_name, train_file_name, train)
    save_data_to_gcs(bucket_name, test_file_name, test)
    
    # Save encoders to GCS
    encoders = {"user_encoder": user_encoder, "product_encoder": product_encoder}
    save_encoders_to_gcs(bucket_name, encoders, encoders_file_name)

    logging.info("Preprocessing complete")

# Run the script
if __name__ == "__main__":
    preprocess()

