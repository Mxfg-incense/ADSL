import json
import pandas as pd
import glob

# Get a list of all JSON files in the ../results/ directory
json_files = glob.glob('../results/*.json')

# Initialize an empty list to store the data
data = []

# Loop over the JSON files
for json_file in json_files:
    # Open the JSON file and load the data
    with open(json_file, 'r') as f:
        data.append(json.load(f))

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Keep only specific columns
df = df[['data_source', 'indep_graph', 'AUC', 'AUPR', 'F1', 'balance_acc']]

# # Set the precision of each table entry to 4 decimal places
# pd.set_option('precision', 4)

# Write the DataFrame to a CSV file
df.to_csv('../results/results.csv', index=False)