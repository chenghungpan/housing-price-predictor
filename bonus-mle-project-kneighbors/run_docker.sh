#!/bin/bash

# Define the app name
APP_NAME="bonus-housing-price-predictor-kneighbors"

# Build the Docker image
echo "Building Docker image..."
docker build -t $APP_NAME .

# Run the Docker container
echo "Running $APP_NAME..."
docker run -p 5000:5000 --name bonus-app $APP_NAME

