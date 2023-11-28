#!/bin/bash

# Define the app name
APP_NAME="housing-price-predictor-kneighbors"

# Build the Docker image
echo "Building Docker image..."
docker build -t $APP_NAME .

# Run the Docker container
echo "Running $APP_NAME..."
docker run -p 5000:5000 --name blue-app $APP_NAME

