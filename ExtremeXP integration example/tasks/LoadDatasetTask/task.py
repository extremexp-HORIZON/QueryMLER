import os, sys
import eexp_engine.executionware.proactive_helper as ph
[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]

import requests
import pandas as pd

print("Running LoadDatasetTask - Loading and uploading datasets")

endpoint_ip = "146.124.106.225"
endpoint_port = 5000
cleanup_hours = 6

# Load publications dataset from CSV file
print("Loading publications dataset...")
publications_path = variables.get("PublicationsInput")
publications_data = pd.read_csv(publications_path, sep='>', on_bad_lines='skip')
print(f"Publications data loaded successfully: {len(publications_data)} records")

# Load ground truth dataset from CSV file
print("Loading ground truth dataset...")
ground_truth_path = variables.get("GroundTruthInput")
ground_truth_data = pd.read_csv(ground_truth_path, sep=';', on_bad_lines='skip')
print(f"Ground truth data loaded successfully: {len(ground_truth_data)} records")

# Convert datasets to CSV strings
publications_csv = publications_data.to_csv(index=False)
ground_truth_csv = ground_truth_data.to_csv(index=False)

endpoint_url = f"http://{endpoint_ip}:{endpoint_port}/upload-dataset"
print(f"Uploading to endpoint: {endpoint_url}")

try:
    print("Uploading publications.csv...")
    files = {
        'file': ('publications.csv', publications_csv, 'text/csv')
    }
    data = {
        'temporary': 'false',
        'cleanup_hours': str(cleanup_hours)
    }
    
    response1 = requests.post(
        endpoint_url,
        files=files,
        data=data
    )
    
    print(f"Upload response status: {response1.status_code}")
    if response1.status_code != 200:
        print(f"Response headers: {response1.headers}")
        print(f"Response body: {response1.text}")
    
    response1.raise_for_status()
    print(f"Publications uploaded successfully. Response: {response1.status_code}")
    
    print("Uploading ground_truth_publications.csv...")
    files = {
        'file': ('ground_truth_publications.csv', ground_truth_csv, 'text/csv')
    }
    
    response2 = requests.post(
        endpoint_url,
        files=files,
        data=data
    )
    
    print(f"Upload response status: {response2.status_code}")
    if response2.status_code != 200:
        print(f"Response headers: {response2.headers}")
        print(f"Response body: {response2.text}")
    
    response2.raise_for_status()
    print(f"Ground truth uploaded successfully. Response: {response2.status_code}")
    
    variables.put("PUBLICATIONS", publications_data.to_json(orient='records'))
    variables.put("GROUND_TRUTH", ground_truth_data.to_json(orient='records'))
    
    resultMap.put("upload_status", "SUCCESS")
    print("Both datasets loaded and uploaded successfully!")
    
except requests.exceptions.RequestException as e:
    print(f"Error uploading datasets: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Server response: {e.response.text}")
    resultMap.put("upload_status", f"FAILED: {str(e)}")
    raise
