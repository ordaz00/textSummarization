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
train_data_list = [json.loads(line) for line in train_data_json.split('\n') if line]
train_data = spark.createDataFrame(train_data_list)

# Assuming 'features_article' is the input_seq and 'features_highlights' is the target_seq
input_sequences = train_data.select("features_article").rdd.map(lambda x: np.array(x[0])).collect()
target_sequences = train_data.select("features_highlights").rdd.map(lambda x: np.array(x[0])).collect()

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