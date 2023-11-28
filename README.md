# Housing Price Prediction System

## Overview
This project is a Housing Price Prediction System that leverages machine learning to provide accurate predictions of housing prices based on various features. It offers a user-friendly API for users to input property details and receive predictions.

## Features
- Utilizes a pre-trained machine learning model for precise predictions.
- Offers a RESTful API for easy integration into other applications.
- Provides metadata and additional information for context.
- There are three subdictories: 
     1) blue-mle-project-kneighbors/ 
     -original k neighbors model with R2 of `73.63%`.
     2) green-mle-project-xgboost/
     -modified Xgboost model with R2 of `86.64%`.
     3) bonus-mle-project-kneighbors/
     -k neighbors model with only minimum features with R2 of `53.66`.
- Blue/Green Deployment: Docker built images and deployment, allow real-time switch between blue app and green app.

## Getting Started
#### 1. Clone the repository
   `git clone <repository-url>`<br>
#### 2. Activate the environment:
`conda env create -f conda_environment.yml`<br>
`conda activate housing`<br>

#### 3. blue_app: 
`cd blue-mle-project-kneighbors`
`./run_docker`
the blue-app will be deployed, and the localhost is ready to serve
you can run in a different window (with conda venv 'housing') under the same directory:
`python test.py`
it will show 100 testing url responses
Also, you evaluate the performance by :<br>
`python evaluate.py` 

#### 4. green_app: 
`cd green-mle-project-xgboost`
`./run_docker`
the blue-app will be deployed, and the localhost is ready to serve
you can run in a different window (with conda venv 'housing') under the same directory: <br>
`python test.py` <br>
it will show 100 testing url responses. <br>
Also, you evaluate the performance by : <br>
`python evaluate.py` <br>
#### 5. bonus_app: 
`cd green-mle-project-xgboost` <br>
`./run_docker` <br>
the blue-app will be deployed, and the localhost is ready to serve
you can run in a different window (with conda venv 'housing') under the same directory: <br>
`python test.py` <br>
it will show 100 testing url responses <br>
Also, you evaluate the performance by :<br>
`python evaluate.py` 

#### 6. blue/green test:
You can start or stop either blue or green tests to control which app will be deployed: <br>
`docker start blue-app` <br>
`python evaluate.py` <br>
`docker stop blue-app` <br>
`docker start green-app` <br>
`python evaluate.py` <br>
`docker stop green-app` <br>

#### 7. bonus test: (go to bonus-mle-project directory)
`docker start bonus-app` <br>
`python test.py` <br>
`python evaluate.py` <br>



