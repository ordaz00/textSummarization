# Build Docker Image
docker build -t your_image_name .

# Tag Docker Image
docker tag text_summarization gcr.io/ornate-woodland-384921/text_summarization

# Push Docker Image
docker push gcr.io/ornate-woodland-384921/text_summarization

# Run Docker Container
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=text_summarization4 \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/ornate-woodland-384921/text_summarization

# Print updates from the cloud
gcloud ai custom-jobs stream-logs projects/184966663018/locations/us-central1/customJobs/2542488010641899520