from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, IDF
from google.cloud import storage
import os

# Correcting the python versions
os.environ["PYSPARK_PYTHON"] = "/Users/ordaz00/miniconda3/envs/pyspark_env/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Users/ordaz00/miniconda3/envs/pyspark_env/bin/python"


# Initialize the spark session
spark = SparkSession.builder \
    .appName("Text Summarization") \
    .config("spark.executor.memory", "14g") \
    .config("spark.driver.memory", "14g") \
    .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
    .config("spark.cleaner.referenceTracking.blocking", "true") \
    .config("spark.cleaner.referenceTracking.blockingFraction", "0.5") \
    .getOrCreate()

# Initializing storage client
storage_client = storage.Client.from_service_account_json('ornate-woodland-384921-a68699295d78.json')

# Column names
text_columns = ["article", "highlights"]
clean_text_columns = ["clean_article", "clean_highlights"]
tokens_columns = ["tokens_article", "tokens_highlights"]
raw_features_columns = ["raw_features_article", "raw_features_highlights"]
features_columns = ["features_article", "features_highlights"]

# Bucket Variables
bucket_name = "dailynews_text_summarization"
train_blob_name = "preprocessed_training.json"
test_blob_name = "preprocessed_test.json"
validation_blob_name = "preprocessed_validation.json"

# Preprocess text
def preprocess_text(data, text_columns, clean_text_columns, tokens_columns):
    for i in range(len(text_columns)):
        data = data.withColumn("lowercase_text", F.lower(F.col(text_columns[i])))
        data = data.withColumn(clean_text_columns[i], F.regexp_replace(F.col("lowercase_text"), "[^a-zA-Z\s]", ""))
        data = data.withColumn(tokens_columns[i], F.split(F.col(clean_text_columns[i]), "\s+"))
        data = data.drop("lowercase_text")
    return data

# Process data
def process_data(data, text_columns, clean_text_columns, tokens_columns, raw_features_columns, features_columns):
    data = preprocess_text(data, text_columns, clean_text_columns, tokens_columns)
    data = data.dropna(subset=tokens_columns)
    for i in range(len(tokens_columns)):
        data = extract_features(data, tokens_columns[i], raw_features_columns[i], features_columns[i])
    return data

# Extract features
def extract_features(data, tokens_column, raw_features_column, features_column):
    hashingTF = HashingTF(inputCol=tokens_column, outputCol=raw_features_column)
    data = hashingTF.transform(data)
    idf = IDF(inputCol=raw_features_column, outputCol=features_column)
    idf_model = idf.fit(data)
    return idf_model.transform(data)

# Function to upload json dataframes to Google Cloud Storage
def upload_json_string(bucket_name, json_data, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Join the JSON data list into a single string with newline separators
    json_string = "\n".join(json_data)
    
    # Upload the JSON string to Google Cloud Storage
    blob.upload_from_string(json_string, content_type='application/json')

def prepareDatasets():
    # Load the datasets
    train_data = spark.read.csv("train.csv", header=True, inferSchema=True)
    test_data = spark.read.csv("test.csv", header=True, inferSchema=True)
    validation_data = spark.read.csv("validation.csv", header=True, inferSchema=True)

    # Repartition the datasets
    train_data = train_data.repartition(200)
    test_data = test_data.repartition(200)
    validation_data = validation_data.repartition(200)

    # Process the datasets
    train_data = process_data(train_data, text_columns, clean_text_columns, tokens_columns, raw_features_columns, features_columns)
    test_data = process_data(test_data, text_columns, clean_text_columns, tokens_columns, raw_features_columns, features_columns)
    validation_data = process_data(validation_data, text_columns, clean_text_columns, tokens_columns, raw_features_columns, features_columns)

def uploadToCloud(train_data, test_data, validation_data):
    # Convert dataframes to json
    train_data_json = train_data.toJSON().saveAsTextFile("train_data_preprocessed")
    test_data_json = test_data.toJSON().collect()
    validation_data_json = validation_data.toJSON().collect()
    # Upload the JSON strings to Google Cloud Storage
    upload_json_string(bucket_name, "train_data_preprocessed", train_blob_name)
    upload_json_string(bucket_name, test_data_json, test_blob_name)
    upload_json_string(bucket_name, validation_data_json, validation_blob_name)
