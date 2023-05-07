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

def read_directory_from_cloud_to_pyspark_df(bucket_name, directory_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=directory_name)
    
    files_contents = []
    
    for blob in blobs:
        if not blob.name.endswith('/'):  # Exclude directories
            file_content = blob.download_as_text()
            file_json = [json.loads(line) for line in file_content.split('\n') if line]
            files_contents.extend(file_json)
    
    return spark.createDataFrame(files_contents)

# Load training data from Google Cloud Storage as a DataFrame
train_data = read_directory_from_cloud_to_pyspark_df(bucket, "train_data_preprocessed")

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

# Simple Seq2Seq model using TensorFlow
encoder_inputs = tf.keras.Input(shape=(None, input_sequences[0].shape[0]))
encoder = tf.keras.layers.LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.Input(shape=(None, target_sequences[0].shape[0]))
decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(target_sequences[0].shape[0], activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([input_sequences, target_sequences], target_sequences, batch_size=64, epochs=100)

# Evaluate the model on validation dataset
validation_loss, validation_accuracy = model.evaluate([validation_input_sequences, validation_target_sequences], validation_target_sequences, batch_size=64)

# Save validation results to a file
validation_results_file = "model_validation.txt"
with open(validation_results_file, 'w') as v:
    v.write(f"Loss: {validation_loss}\n")
    v.write(f"Accuracy: {validation_accuracy}\n")

# Evaluate the model on test dataset
test_loss, test_accuracy = model.evaluate([test_input_sequences, test_target_sequences], test_target_sequences, batch_size=64)

# Save evaluation results to a file
test_results_file = "model_test.txt"
with open(test_results_file, 'w') as t:
    t.write(f"Loss: {test_loss}\n")
    t.write(f"Accuracy: {test_accuracy}\n")

# Save the model as an H5 file
model.save("trained_model.h5")

# Upload validation, test, and model files to Google Cloud
store_validation_data = bucket.blob("model_validation.txt")
store_test_data = bucket.blob("model_test.txt")
store_model = bucket.blob("trained_model.h5")

store_validation_data.upload_from_filename("model_validation.txt")
store_test_data.upload_from_filename("model_test.txt")
store_model.upload_from_filename("trained_model.h5")
