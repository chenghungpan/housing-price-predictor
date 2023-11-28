import requests
import json
import pandas as pd

# Define the API endpoint URL
url = 'http://localhost:5000/predict'  # Replace with your API URL if different

# Load the model_features.json file to determine feature names and order
with open("model/model_features.json", "r") as feature_file:
    features = json.load(feature_file)

# Load the data from the CSV file containing future unseen examples
data_future = pd.read_csv("data/future_unseen_examples.csv",
                                    dtype={'zipcode': str})


# Define a list of feature names to include from the CSV data
# Exclude the 'price', 'date', 'id', and demographic data columns
selected_features = [feature for feature in features if feature not in ('price', 'date', 'id')]

# Load the data from the CSV file containing kc_house_data
data_kc_house = pd.read_csv("data/kc_house_data.csv", dtype={'zipcode': str})

# # Merge the future unseen examples data with kc_house_data on 'zipcode' and drop unnecessary columns
# merged_data = data_future.merge(data_kc_house, on='zipcode', how='left').drop(columns=['price', 'date', 'id'])
# # merged_data = data_kc_house.merge(data_future, on='zipcode', how='left').drop(columns=['price', 'date', 'id'])

# Remove duplicates by aggregating with first()
aggregated_data_kc_house = data_kc_house.groupby('zipcode').first()

# Step 1: Merge with suffixes
merged_data = data_future.merge(aggregated_data_kc_house, on='zipcode', how='left', suffixes=('_left', '_right'))

# Step 2: Drop columns from the right DataFrame
cols_to_drop = [col for col in merged_data.columns if col.endswith('_right')]
merged_data.drop(cols_to_drop, axis=1, inplace=True)

# Optional: Rename columns to remove the '_left' suffix
merged_data.columns = [col.replace('_left', '') for col in merged_data.columns]

# Load the data from the CSV file containing zipcode_demographics
data_demographics = pd.read_csv("data/zipcode_demographics.csv",
                                    dtype={'zipcode': str})

# Merge the merged_data with zipcode_demographics on 'zipcode'
final_data_x = merged_data.merge(data_demographics, 
                    on='zipcode', how='left')

final_data = final_data_x[selected_features]


# Iterate through each row of the final_data and create input JSON
input_data_list = []

for _, row in final_data.iterrows():
    input_data = {feature: row[feature] for feature in final_data}
    input_data_list.append(input_data)
input_df = pd.DataFrame(input_data_list)



# # Send POST requests for each input data
cnt=0
for index, row in input_df.iterrows():
    cnt+=1

    input_item = pd.DataFrame([row])
    input_json=input_item.to_json(orient='records')
    # print('input_json=', input_json)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=input_json, headers = headers)
    print(cnt,'response=', response.json())
  
    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # print('data=', data)

        # Extract the prediction and metadata from the response
        prediction = data.get('prediction', 'Prediction not available')
        metadata = data.get('metadata', 'Metadata not available')

        # Print the prediction and metadata
        # print(f'{cnt}: Prediction: {prediction}')
        # print(f'Metadata: {metadata}')
    else:
        print(f'Error: HTTP status code {response.status_code}')
        print(response.text)  # Print the error message if available
