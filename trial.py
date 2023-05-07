import tensorflow as tf
from google.cloud import storage
import json
import numpy as np
from pyspark.sql import SparkSession

# Set up Google Cloud Storage client
JSON_KEYFILE_PATH = 'ornate-woodland-384921-a68699295d78.json'
storage_client = storage.Client.from_service_account_json(JSON_KEYFILE_PATH)
bucket = storage_client.bucket("dailynews_text_summarization")

# Initialize the spark session (if not already done)
spark = SparkSession.builder \
    .appName("Text Summarization") \
    .config("spark.executor.memory", "14g") \
    .config("spark.driver.memory", "14g") \
    .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
    .config("spark.cleaner.referenceTracking.blocking", "true") \
    .config("spark.cleaner.referenceTracking.blockingFraction", "0.5") \
    .getOrCreate()

# Load training data from Google Cloud Storage as a DataFrame
train_blob_name = "preprocessed_training.json"
train_data_blob = bucket.blob(train_blob_name)
train_data_json = train_data_blob.download_as_text()
print(train_data_json[:500])
# train_data_list = [json.loads(line) for line in train_data_json.split('\n') if line]
train_data_list = []

for line in train_data_json.split('\n'):
    if not line:
        continue
    try:
        train_data_list.append(json.loads(line))
    except json.JSONDecodeError as e:
        print(f"Error occurred while decoding: {e}")
        print(f"Offending line: {line}")
        raise
train_data = spark.createDataFrame(train_data_list)

# Load validation data from Google Cloud Storage as a DataFrame
validation_blob_name = "preprocessed_validation.json"
validation_data_blob = bucket.blob(validation_blob_name)
validation_data_json = validation_data_blob.download_as_text()
validation_data_list = [json.loads(line) for line in validation_data_json.split('\n') if line]
validation_data = spark.createDataFrame(validation_data_list)

# Load test data from Google Cloud Storage as a DataFrame
test_blob_name = "preprocessed_test.json"
test_data_blob = bucket.blob(test_blob_name)
test_data_json = test_data_blob.download_as_text()
test_data_list = [json.loads(line) for line in test_data_json.split('\n') if line]
test_data = spark.createDataFrame(test_data_list)

# Assuming 'features_article' is the input_seq and 'features_highlights' is the target_seq
input_sequences = train_data.select("features_article").rdd.map(lambda x: np.array(x[0])).collect()
target_sequences = train_data.select("features_highlights").rdd.map(lambda x: np.array(x[0])).collect()

# Create input and target sequences for the validation dataset
validation_input_sequences = validation_data.select("features_article").rdd.map(lambda x: np.array(x[0])).collect()
validation_target_sequences = validation_data.select("features_highlights").rdd.map(lambda x: np.array(x[0])).collect()

# Create input and target sequences for the test dataset
test_input_sequences = test_data.select("features_article").rdd.map(lambda x: np.array(x[0])).collect()
test_target_sequences = test_data.select("features_highlights").rdd.map(lambda x: np.array(x[0])).collect()