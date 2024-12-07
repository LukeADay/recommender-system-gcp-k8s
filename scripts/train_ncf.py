import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
import pandas as pd
from google.cloud import storage
import logging
import io  # Import the correct module for StringIO


def load_data_from_gcs(bucket_name, file_name):
    """
    Load CSV data from a Google Cloud Storage bucket.
    """
    logging.info(f"Loading data from GCS: {bucket_name}/{file_name}")
    client = storage.Client(project="recommender-system-gcp-k8s")  # Explicitly set the project
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_text()
    return pd.read_csv(io.StringIO(data))  # Ensure io.StringIO is used here


def normalize_ids(data, id_column):
    """
    Normalize IDs to be zero-indexed and continuous.
    """
    unique_ids = data[id_column].unique()
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    data[id_column] = data[id_column].map(id_map)
    return data, len(unique_ids)


# Build the NCF model
def build_ncf_model(n_users, n_products, embedding_dim=50):
    """
    Build the Neural Collaborative Filtering (NCF) model.
    """
    user_input = Input(shape=(1,), name="user_input")
    user_embedding = Embedding(n_users, embedding_dim, name="user_embedding")(user_input)
    user_vec = Flatten()(user_embedding)

    product_input = Input(shape=(1,), name="product_input")
    product_embedding = Embedding(n_products, embedding_dim, name="product_embedding")(product_input)
    product_vec = Flatten()(product_embedding)

    concat = Concatenate()([user_vec, product_vec])
    fc = Dense(128, activation="relu")(concat)
    fc = Dense(64, activation="relu")(fc)
    output = Dense(1, activation="sigmoid")(fc)

    model = Model(inputs=[user_input, product_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Save model to GCS
def save_model_to_gcs(model, bucket_name, model_dir):
    """
    Save the trained model to Google Cloud Storage.
    """
    logging.info(f"Saving model to GCS: {bucket_name}/{model_dir}")
    local_model_dir = "/tmp/ncf_model"  # Temporary local directory
    model.save(local_model_dir)  # Save the model locally first

    # Upload the saved model to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_dir)
    blob.upload_from_filename(f"{local_model_dir}/saved_model.pb")


# Train the model
def train():
    """
    Train the Neural Collaborative Filtering (NCF) model.
    """
    # GCS paths
    bucket_name = "recommender-system-bucket"
    train_data_path = "processed/train.csv"
    test_data_path = "processed/test.csv"

    # Load train/test data
    train = load_data_from_gcs(bucket_name, train_data_path)
    test = load_data_from_gcs(bucket_name, test_data_path)

    # Normalize IDs
    logging.info("Normalizing IDs for training and test data")
    train, n_users = normalize_ids(train, "user_id")
    train, n_products = normalize_ids(train, "product_id")

    test, _ = normalize_ids(test, "user_id")
    test, _ = normalize_ids(test, "product_id")

    # Extract train/test inputs
    train_users = train['user_id'].values
    train_products = train['product_id'].values
    train_labels = train['interaction_strength'].values

    test_users = test['user_id'].values
    test_products = test['product_id'].values
    test_labels = test['interaction_strength'].values

    # Build and train model
    logging.info("Building and training the NCF model")
    model = build_ncf_model(n_users, n_products)

    model.fit(
        [train_users, train_products], train_labels,
        validation_data=([test_users, test_products], test_labels),
        epochs=10,
        batch_size=256
    )

    # Save the trained model
    save_model_to_gcs(model, bucket_name, "processed/ncf_model")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
