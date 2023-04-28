from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# Initialize the spark session
spark = SparkSession.builder \
    .appName("Text Summarization") \
    .getOrCreate()

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

# Load the datasets
train_data = spark.read.csv("train.csv", header=True, inferSchema=True)
test_data = spark.read.csv("test.csv", header=True, inferSchema=True)
validation_data = spark.read.csv("validation.csv", header=True, inferSchema=True)

# Column names
text_columns = ["article", "highlights"]
clean_text_columns = ["clean_article", "clean_highlights"]
tokens_columns = ["tokens_article", "tokens_highlights"]
raw_features_columns = ["raw_features_article", "raw_features_highlights"]
features_columns = ["features_article", "features_highlights"]

# Process the datasets
train_data = process_data(train_data, text_columns, clean_text_columns, tokens_columns, raw_features_columns, features_columns)
test_data = process_data(test_data, text_columns, clean_text_columns, tokens_columns, raw_features_columns, features_columns)
validation_data = process_data(validation_data, text_columns, clean_text_columns, tokens_columns, raw_features_columns, features_columns)


