# Build Docker Image
docker build --no-cache -t textsummarization .

#docker build -t text_summarization.

# Tag Docker Image
docker tag textsummarization gcr.io/ornate-woodland-384921/textsummarization

# Push Docker Image
docker push gcr.io/ornate-woodland-384921/textsummarization

# Run Docker Container
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=textsummarization \
  --worker-pool-spec=machine-type=n1-highmem-4,replica-count=1,container-image-uri=gcr.io/ornate-woodland-384921/textsummarization,accelerator-type=NVIDIA_TESLA_K80,accelerator-count=1
