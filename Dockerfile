FROM python:3.9

# Install necessary dependencies
RUN pip install tensorflow google-cloud-storage pyspark

# Set the working directory
WORKDIR /app

# Copy the python file and JSON keyfile to the working directory
COPY textSummarization.py /app/your_script.py
COPY ornate-woodland-384921-a68699295d78.json /app/ornate-woodland-384921-a68699295d78.json

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ornate-woodland-384921-a68699295d78.json

# Run the script
CMD ["python", "your_script.py"]
