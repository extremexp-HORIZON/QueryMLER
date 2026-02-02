import os, sys
import eexp_engine.executionware.proactive_helper as ph
[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]

import requests
import json

print("Running DataIntegrationTask - Performing inference")

# Load training results from variables
training_results_json = variables.get("TrainingResults")
training_results = json.loads(training_results_json)
print(f"Training results loaded: {training_results}")

endpoint_ip = "146.124.106.225"
endpoint_port = 5000
dataset_name = "publications"
query = "SELECT DEDUP * FROM publications WHERE MOD(id, 10) = 1"
csv_delimiter = ","
model_name = "/app/models/my_distilbert_model"
tokenizer_name = "/app/models/my_distilbert_tokenizer"
batch_size = 128
max_seq_length = 128
confidence_threshold = 0
output_format = "pairs"
# output_format = "deduplicated"
endpoint_url = f"http://{endpoint_ip}:{endpoint_port}/inference"
print(f"Inference endpoint: {endpoint_url}")

try:
    print("Preparing inference payload...")
    inference_payload = {
        "dataset_name": dataset_name,
        "query": query,
        "csv_delimiter": csv_delimiter,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "confidence_threshold": confidence_threshold,
        "output_format": output_format
    }
    
    print(f"Starting inference on dataset...")
    print(f"Query: {query}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    response = requests.post(
        endpoint_url,
        json=inference_payload,
        headers={"Content-Type": "application/json"},
        timeout=600
    )
    response.raise_for_status()
    
    response_data = response.json()
    print(f"Inference completed successfully!")
    print(f"Response: {json.dumps(response_data, indent=2)}")
    
    duplicate_count = response_data.get("duplicate_pairs_count", 0)
    if duplicate_count:
        resultMap.put("duplicate_pairs_found", int(duplicate_count))
        print(f"Duplicate pairs found: {duplicate_count}")
    
    resultMap.put("inference_status", "SUCCESS")
    
    # Store inference results as JSON string for downstream tasks
    variables.put("InferenceResults", json.dumps(response_data))
    
    print("Data integration inference task completed successfully!")
    
except requests.exceptions.Timeout:
    print("Inference request timed out after 300 seconds")
    resultMap.put("inference_status", "TIMEOUT")
    raise
except requests.exceptions.RequestException as e:
    print(f"Error during inference: {e}")
    resultMap.put("inference_status", f"FAILED: {str(e)}")
    raise
