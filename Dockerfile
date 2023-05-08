# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Install Python 3.9 and other dependencies
RUN apt-get update && \
    apt-get install -y python3.9 python3-pip openjdk-11-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3.9 -m pip install --upgrade pip

# Install necessary dependencies
RUN python3.9 -m pip install tensorflow-gpu==2.6.0 google-cloud-storage pyspark

# Set the working directory
WORKDIR /app

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

# Copy the python file and JSON keyfile to the working directory
COPY textSummarization.py /app/textSummarization.py
COPY ornate-woodland-384921-a68699295d78.json /app/ornate-woodland-384921-a68699295d78.json

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ornate-woodland-384921-a68699295d78.json

# Run the script
CMD ["python3.9", "textSummarization.py"]
