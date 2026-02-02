import os, sys
import eexp_engine.executionware.proactive_helper as ph
[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]

import requests
import json

print("Running FineTuneLLMTask - Training model")

endpoint_ip = "146.124.106.225"
endpoint_port = 5000
dataset_name = "publications"
query = "SELECT DEDUP * FROM publications WHERE MOD(id, 10) = 0"
ground_truth_path = "/tmp/ground_truth_publications.csv"
ground_truth_delimiter = ","
csv_delimiter = ","
model = "prajjwal1/bert-tiny"
epochs = 5
batch_size = 128
learning_rate = 0.00005
max_seq_length = 128
evaluation_metric = "f1_score"
model_name = "/app/models/my_distilbert_model"
tokenizer_name = "/app/models/my_distilbert_tokenizer"

endpoint_url = f"http://{endpoint_ip}:{endpoint_port}/train"
print(f"Training endpoint: {endpoint_url}")

try:
    print("Preparing training payload...")
    train_payload = {
        "dataset_name": dataset_name,
        "query": query,
        "ground_truth": ground_truth_path,
        "ground_truth_delimiter": ground_truth_delimiter,
        "csv_delimiter": csv_delimiter,
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "evaluation_metric": evaluation_metric,
        "class_weights": [1.0, 5.0],
        "model_name": model_name,
        "tokenizer_name": tokenizer_name
    }
    
    print(f"Starting model training with {epochs} epochs...")
    print(f"Query: {query}")
    print(f"Model: {model}")
    
    response = requests.post(
        endpoint_url,
        json=train_payload,
        headers={"Content-Type": "application/json"},
        timeout=600
    )
    response.raise_for_status()
    
    response_data = response.json()
    print(f"Training completed successfully!")
    print(f"Response: {json.dumps(response_data, indent=2)}")
    
    f1_score = response_data.get("f1_score", 0.0)
    if f1_score:
        resultMap.put("training_f1_score", float(f1_score))
        print(f"Training F1 Score: {f1_score}")
    
    resultMap.put("training_status", "SUCCESS")
    
    # Store training results as JSON string for downstream tasks
    variables.put("TrainingResults", json.dumps(response_data))
    
    print("Model training task completed successfully!")
    
except requests.exceptions.Timeout:
    print("Training request timed out after 600 seconds")
    resultMap.put("training_status", "TIMEOUT")
    raise
except requests.exceptions.RequestException as e:
    print(f"Error during model training: {e}")
    resultMap.put("training_status", f"FAILED: {str(e)}")
    raise
