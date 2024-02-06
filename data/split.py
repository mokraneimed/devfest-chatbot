import json
import pandas as pd

file_path = 'data/devfest_data.csv'
data = pd.read_csv(file_path).to_dict()

chunks = []
for i in range(len(data['scope'])):
    chunk = f"{data['scope'][i]}\n{data['title'][i]}\n{data['detail'][i]}"
    chunks.append(chunk)
    
with open('data/devfest_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=4)