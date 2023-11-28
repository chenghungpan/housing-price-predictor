import json
import requests
import pathlib
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics."""
    data = pd.read_csv(sales_path,
                       usecols=sales_column_selection,
                       dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged_data.pop('price')
    x = merged_data

    return x, y

def main():
    """Load data, train model, evaluate and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

    print('number of test samples:', len(x_test))

    # Define the API endpoint URL
    url = 'http://localhost:5000/predict'  # Replace with your API URL if different
    headers = {'Content-Type': 'application/json'}

    predictions = []
    cnt=0
    for index, row in x_test.iterrows():
        cnt+=1
        input_item = pd.DataFrame([row])
        input_json=input_item.to_json(orient='records')
        # print('input_json=', input_json)
        response = requests.post(url, data=input_json, headers = headers)


            # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract the prediction and metadata from the response
            prediction = data.get('prediction', 'Prediction not available')
            metadata = data.get('metadata', 'Metadata not available')

            # Print the prediction and metadata
            # print(f'{cnt}: Prediction: {prediction}')
            if cnt%1000==0:
                print(cnt)
            # print(f'Metadata: {metadata}')

            predictions.append(prediction)
        else:
            print(f'Error: HTTP status code {response.status_code}')
            print(response.text)  # Print the error message if available

    y_pred = pd.DataFrame(predictions, columns=['price'])



    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared: {r2}')


if __name__ == "__main__":
    main()

