import json
from constants import *

with open(f'{REPO_ROOT_PATH}/chrissu/system_outputs/system_output_2.json', 'r') as f:
    data = json.load(f)

filtered_data = {
    key: value for key, value in data.items() 
    if value == "" or value == "Not found."
}

print(json.dumps(filtered_data, indent=2))

with open('filtered_results.json', 'w') as f:
    json.dump(filtered_data, f, indent=2)
