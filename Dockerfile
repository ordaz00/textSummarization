FROM python:3.9

# Install necessary dependencies
RUN pip install tensorflow google-cloud-storage pyspark

# Set the working directory
WORKDIR /app

# Install Java
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

# Copy the python file and JSON keyfile to the working directory
COPY textSummarization.py /app/textSummarization.py
COPY ornate-woodland-384921-a68699295d78.json /app/ornate-woodland-384921-a68699295d78.json

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ornate-woodland-384921-a68699295d78.json

# Run the script
CMD ["python", "textSummarization.py"]
