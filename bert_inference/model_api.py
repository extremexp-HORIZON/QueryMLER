from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import requests
import subprocess
import time
import uuid
import os
import threading

def create_train_dataset(dataset, ground_truth, query, dataset_name, train_df_name, ground_truth_delimiter=';', csv_delimiter='>'):
    
    # Generate unique candidate filename to avoid race conditions with concurrent requests
    request_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    candidates_file = f'/data/candidates_{request_id}.csv'
    
    print(f"Using unique candidates file: {candidates_file}")
    
    load_dataset_url = "http://api:8080/api/load-dataset"
    load_payload = {
        "datasetPath": f"/data/{dataset_name}.csv",
        "datasetName": dataset_name,
        "schemaName": "csv",
        "temporary": True  # Mark as temporary for automatic cleanup after 12 hours
    }
    
    print(f"Loading dataset into Java API: {dataset_name}")
    try:
        load_response = requests.post(load_dataset_url, json=load_payload, timeout=300)
        if load_response.status_code == 200:
            print(f"Dataset loaded successfully: {load_response.json().get('message', 'OK')}")
        else:
            print(f"WARNING: Dataset load returned status {load_response.status_code}")
            print(f"Response: {load_response.text}")
    except Exception as e:
        print(f"WARNING: Failed to load dataset in Java API: {e}")
        print("Continuing anyway - dataset may already be loaded or will be loaded on startup")
    
    url = "http://api:8080/api/query"
    params = {
        "q": query,
        "page": 0,
        "offset": -1,  # Get all results
        "candidatesFile": candidates_file  # Pass the unique filename to Java API
    }
    
    print(f"Calling Java API with query: {query}")
    response = requests.post(url, params=params, timeout=120)
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        # DEDUP query writes candidates to unique file specified in params
        
        try:
            candidates = pd.read_csv(candidates_file)
            print(f"Loaded {len(candidates)} candidate pairs from {candidates_file}")
        except FileNotFoundError:
            print(f"WARNING: {candidates_file} not found")
            candidates = pd.DataFrame(columns=['id1', 'id2'])
        
        # Clean up the candidates file after reading
        try:
            if os.path.exists(candidates_file):
                os.remove(candidates_file)
                print(f"Cleaned up candidates file: {candidates_file}")
        except Exception as e:
            print(f"WARNING: Could not delete {candidates_file}: {e}")
        
        # Load ground truth with configurable delimiter
        print(f"Loading ground truth (delimiter: '{ground_truth_delimiter}')")
        ground_truth_df = pd.read_csv(ground_truth, sep=ground_truth_delimiter, names=['id1', 'id2'])
        print(f"Ground truth: {len(ground_truth_df)} pairs")
        
        print(f"Reading dataset from CSV...")
        dataset_csv = f'/data/{dataset_name}.csv'
        all_records = pd.read_csv(dataset_csv, sep=csv_delimiter, header=0, on_bad_lines='skip')
        print(f"Loaded {len(all_records)} records")
        
        if all_records is not None and len(all_records) > 0:
            
            # Create a lookup dict for fast access
            records_dict = {str(row['id']): row for _, row in all_records.iterrows()}
            
            candidate_pairs = set((str(row['id1']), str(row['id2'])) for _, row in candidates.iterrows())
            ground_truth_pairs_set = set((str(row['id1']), str(row['id2'])) for _, row in ground_truth_df.iterrows())
            overlap = len(candidate_pairs.intersection(ground_truth_pairs_set))
            print(f"Overlap with ground truth: {overlap} pairs")
            
            positive_samples = []
            negative_samples = []
            
            for _, row in candidates.iterrows():
                id1, id2 = str(row['id1']), str(row['id2'])
                pair = (id1, id2)
                
                # Get text for both records
                rec1 = records_dict.get(id1)
                rec2 = records_dict.get(id2)
                
                if rec1 is not None and rec2 is not None:
                    text1 = f"{rec1.get('title', '')} {rec1.get('authors', '')}"
                    text2 = f"{rec2.get('title', '')} {rec2.get('authors', '')}"
                    
                    # Check if this pair is in ground truth
                    is_match = pair in ground_truth_pairs_set or (id2, id1) in ground_truth_pairs_set
                    
                    sample = {'s1': text1, 's2': text2, 'label': is_match}
                    if is_match:
                        positive_samples.append(sample)
                    else:
                        negative_samples.append(sample)
            
            print(f"Created {len(positive_samples)} positive samples from candidates")
            print(f"Created {len(negative_samples)} negative samples from candidates")
            
            if len(positive_samples) > 0 and len(negative_samples) > len(positive_samples) * 2:
                import random
                random.seed(42)  
                negative_samples = random.sample(negative_samples, len(positive_samples) * 2)
                print(f"Downsampled negatives to {len(negative_samples)} (2x positives)")
            
            all_samples = positive_samples + negative_samples
            train_df = pd.DataFrame(all_samples)
            
            print(f"Final training dataset: {len(train_df)} total samples")
            print(f"  - Positive: {(train_df['label'] == True).sum()}")
            print(f"  - Negative: {(train_df['label'] == False).sum()}")
            
            train_df.to_csv(train_df_name, index=False)
            print(f"Saved training data to: {train_df_name}")
        else:
            raise Exception(f"No publication records found in {dataset_csv}")
    else:
        error_msg = f"Java API query failed with status {response.status_code}"
        print(error_msg)
        raise Exception(error_msg)


def train_and_evaluate(train_csv, model, epochs, batch_size,
                       learning_rate, max_seq_length, evaluation_metric,
                       confidence_threshold, top_k_predictions,
                       model_name, tokenizer_name,
                       class_weights=None, loss_func_type="CrossEntropyLoss"):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    use_amp = torch.cuda.is_available()  # Automatic Mixed Precision for speed
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Using device: {device}, Mixed Precision: {use_amp}")

    train_df = pd.read_csv(train_csv)
    train_df = train_df.dropna().reset_index(drop=True)
    # Convert label to int (handles both boolean and string "True"/"False")
    train_df['label'] = train_df['label'].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0, 'true': 1, 'false': 0})
    train_df = train_df.dropna().reset_index(drop=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    train_texts = train_df[['s1', 's2']].values.tolist()
    val_texts = val_df[['s1', 's2']].values.tolist()

    train_labels = train_df['label'].values
    val_labels = val_df['label'].values

    if model == "prajjwal1/bert-tiny": 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        train_encoded_dict = tokenizer.batch_encode_plus(train_texts, max_length=max_seq_length, padding='max_length', truncation=True)
        val_encoded_dict = tokenizer.batch_encode_plus(val_texts, max_length=max_seq_length, padding='max_length', truncation=True)


        train_input_ids = torch.tensor(train_encoded_dict['input_ids'])
        train_attention_masks = torch.tensor(train_encoded_dict['attention_mask'])
        train_labels = torch.tensor(train_labels)

        val_input_ids = torch.tensor(val_encoded_dict['input_ids'])
        val_attention_masks = torch.tensor(val_encoded_dict['attention_mask'])
        val_labels = torch.tensor(val_labels)

        # DataLoader for training data
        train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        # Model init
        model = AutoModelForSequenceClassification.from_pretrained(
        model,
            num_labels=2, 
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(device)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights)
            class_weights = class_weights.to(device)
        else:
            class_weights = None

        if loss_func_type == "CrossEntropyLoss":
            loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif loss_func_type == "BCEWithLogitsLoss":
            loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            raise ValueError(f"Unsupported loss function type: {loss_func_type}")

        loss_func.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

        # Training loop
        for epoch_i in range(0, epochs):
            print("Epoch:", epoch_i + 1)
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_attention_masks = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average Training Loss: {:.4f}".format(avg_train_loss))

        # Validation
        model.eval()
        val_preds, val_true_labels = [], []

        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_attention_masks)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            val_preds.extend(np.argmax(logits, axis=1).flatten())
            val_true_labels.extend(label_ids)

        # Calculate evaluation metric
        if evaluation_metric == "accuracy":
            eval_metric = accuracy_score(val_true_labels, val_preds)
        elif evaluation_metric == "f1_score":
            eval_metric = f1_score(val_true_labels, val_preds)
        elif evaluation_metric == "precision":
            eval_metric = precision_score(val_true_labels, val_preds)
        elif evaluation_metric == "recall":
            eval_metric = recall_score(val_true_labels, val_preds)
        else:
            raise ValueError(f"Unsupported evaluation metric: {evaluation_metric}")

        print(f'Validation {evaluation_metric.capitalize()}: {eval_metric}')

        print(f"Saving model and tokenizer...")
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_name)
        print(f"Model saved successfully")
            
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        train_encoded_dict = tokenizer.batch_encode_plus(train_texts, max_length=max_seq_length, padding='max_length', truncation=True)
        val_encoded_dict = tokenizer.batch_encode_plus(val_texts, max_length=max_seq_length, padding='max_length', truncation=True)

        train_input_ids = torch.tensor(train_encoded_dict['input_ids'])
        train_attention_masks = torch.tensor(train_encoded_dict['attention_mask'])
        train_labels = torch.tensor(train_labels)

        val_input_ids = torch.tensor(val_encoded_dict['input_ids'])
        val_attention_masks = torch.tensor(val_encoded_dict['attention_mask'])
        val_labels = torch.tensor(val_labels)

        # DataLoader for training data
        train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        # Model init
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=2, 
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(device)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights)
            class_weights = class_weights.to(device)
        else:
            class_weights = None

        if loss_func_type == "CrossEntropyLoss":
            loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif loss_func_type == "BCEWithLogitsLoss":
            loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            raise ValueError(f"Unsupported loss function type: {loss_func_type}")

        loss_func.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

        # Training loop
        for epoch_i in range(0, epochs):
            print("Epoch:", epoch_i + 1)
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_attention_masks = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average Training Loss: {:.4f}".format(avg_train_loss))

        # Validation
        model.eval()
        val_preds, val_true_labels = [], []

        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_attention_masks)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            val_preds.extend(np.argmax(logits, axis=1).flatten())
            val_true_labels.extend(label_ids)

        # Calculate evaluation metric
        if evaluation_metric == "accuracy":
            eval_metric = accuracy_score(val_true_labels, val_preds)
        elif evaluation_metric == "f1_score":
            eval_metric = f1_score(val_true_labels, val_preds)
        elif evaluation_metric == "precision":
            eval_metric = precision_score(val_true_labels, val_preds)
        elif evaluation_metric == "recall":
            eval_metric = recall_score(val_true_labels, val_preds)
        else:
            raise ValueError(f"Unsupported evaluation metric: {evaluation_metric}")

        print(f'Validation {evaluation_metric.capitalize()}: {eval_metric}')

        model.save_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_name)        

app = Flask(__name__)

temporary_files = {}

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    """
    Upload a CSV dataset file to the /data volume.
    Marks it as temporary for auto-cleanup after specified hours.
    
    Request body (JSON):
    {
        "filename": "my_dataset.csv",
        "data": "id,title,authors,venue,year\\n1,Paper1,Author1,Venue1,2020\\n...",
        "temporary": true,
        "cleanup_hours": 12
    }
    
    Or multipart/form-data with file upload
    """
    try:
        print("Received dataset upload request.")
        
        # Handle JSON upload
        if request.is_json:
            data = request.get_json()
            filename = data.get('filename')
            file_data = data.get('data')
            temporary = data.get('temporary', True)
            cleanup_hours = data.get('cleanup_hours', 12)
            
            if not filename or not file_data:
                return jsonify({
                    "status": "error",
                    "message": "filename and data are required"
                }), 400
            
            # Save ground truth files to /tmp to avoid Java API scanning them
            if 'ground' in filename.lower() or 'truth' in filename.lower():
                filepath = f'/tmp/{filename}'
            else:
                filepath = f'/data/{filename}'
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(file_data)
                
        # Handle multipart file upload
        elif 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            temporary = request.form.get('temporary', 'true').lower() == 'true'
            cleanup_hours = int(request.form.get('cleanup_hours', 12))
            
            if not filename:
                return jsonify({
                    "status": "error",
                    "message": "No filename provided"
                }), 400
            
            # Save ground truth files to /tmp to avoid Java API scanning them
            if 'ground' in filename.lower() or 'truth' in filename.lower():
                filepath = f'/tmp/{filename}'
            else:
                filepath = f'/data/{filename}'
            
            file.save(filepath)
        else:
            return jsonify({
                "status": "error",
                "message": "No data provided. Use JSON with 'data' field or multipart form with 'file'"
            }), 400
        
        # Track as temporary if requested
        if temporary:
            temporary_files[filepath] = {
                'created_at': time.time(),
                'cleanup_hours': cleanup_hours,
                'filename': filename
            }
            print(f"Marked {filename} as temporary (cleanup after {cleanup_hours} hours)")
        
        print(f"Saved dataset to: {filepath}")
        
        return jsonify({
            "status": "success",
            "message": f"Dataset uploaded successfully",
            "filepath": filepath,
            "filename": filename,
            "temporary": temporary,
            "cleanup_hours": cleanup_hours if temporary else None
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Upload error: {error_details}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500

@app.route('/cleanup-temp-files', methods=['POST'])
def cleanup_temp_files():
    """
    Manually trigger cleanup of old temporary files.
    Automatically called periodically, but can be triggered manually.
    """
    try:
        current_time = time.time()
        cleaned = []
        
        for filepath, info in list(temporary_files.items()):
            age_hours = (current_time - info['created_at']) / 3600
            
            if age_hours >= info['cleanup_hours']:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        cleaned.append({
                            'filepath': filepath,
                            'filename': info['filename'],
                            'age_hours': round(age_hours, 2)
                        })
                        print(f"Cleaned up old file: {filepath} (age: {age_hours:.2f} hours)")
                    del temporary_files[filepath]
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")
        
        return jsonify({
            "status": "success",
            "message": f"Cleaned up {len(cleaned)} temporary files",
            "cleaned_files": cleaned
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        print("Received training request.")
        # Parse params
        data = request.get_json()

        dataset = data.get('dataset')        
        query = data.get('query')
        ground_truth = data.get('ground_truth', '/tmp/ground_truth_publications.csv')  # Default to /tmp
        ground_truth_delimiter = data.get('ground_truth_delimiter', ';')  # Delimiter for ground truth CSV
        csv_delimiter = data.get('csv_delimiter', '>')  # Delimiter for publications CSV
        dataset_name = data.get('dataset_name')

        train_csv_input = data.get('train_csv', '/data/train_publications.csv')
        if train_csv_input.startswith('/data/'):
            train_csv = train_csv_input.replace('/data/', '/tmp/', 1)
        else:
            train_csv = train_csv_input
        
        model = data.get('model', 'distilbert-base-uncased')
        epochs = data.get('epochs', 1)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        max_seq_length = data.get('max_seq_length', 128)
        evaluation_metric = data.get('evaluation_metric', 'accuracy')
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k_predictions = data.get('top_k_predictions', 3)
        class_weights = data.get('class_weights', [1.0, 5.0])
        loss_func_type = data.get('loss_func_type', 'CrossEntropyLoss')
        model_name = data.get('model_name')
        tokenizer_name = data.get('tokenizer_name')

        # Create train dataset
        create_train_dataset(dataset, ground_truth, query, dataset_name, train_csv, ground_truth_delimiter, csv_delimiter)

        # Train
        train_and_evaluate(train_csv, model, epochs, batch_size,
                           learning_rate, max_seq_length, evaluation_metric,
                           confidence_threshold, top_k_predictions,
                           model_name, tokenizer_name,
                           class_weights=class_weights, loss_func_type=loss_func_type)

        return jsonify({"status": "success", "message": "Training completed successfully"})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Training error: {error_details}")
        return jsonify({"status": "error", "message": str(e), "details": error_details})

@app.route('/inference', methods=['POST'])
def inference():
    try:
        print("Received inference request.")
        # Parse params
        data = request.get_json()
        
        query = data.get('query')
        dataset = data.get('dataset')
        dataset_name = data.get('dataset_name')
        csv_delimiter = data.get('csv_delimiter', '>')  # Delimiter for publications CSV
        model_name = data.get('model_name', './my_distilbert_model')
        tokenizer_name = data.get('tokenizer_name', './my_distilbert_tokenizer')
        batch_size = data.get('batch_size', 32)
        max_seq_length = data.get('max_seq_length', 128)
        confidence_threshold = data.get('confidence_threshold', 0.5)
        output_format = data.get('output_format', 'pairs')  # 'pairs' or 'deduplicated'
        
        request_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        candidates_file = f'/data/candidates_{request_id}.csv'
        print(f"Using unique candidates file: {candidates_file}")
        
        print(f"Running query: {query}")
        url = "http://api:8080/api/query"
        params = {
            "q": query,
            "page": 0,
            "offset": -1,  # Get all results
            "candidatesFile": candidates_file
        }
        response = requests.post(url, params=params, timeout=120)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            raise Exception(f"Java API query failed with status {response.status_code}")
        
        try:
            candidates = pd.read_csv(candidates_file)
            print(f"Got {len(candidates)} candidate pairs from {candidates_file}")
            print(f"Candidates columns: {candidates.columns.tolist()}")
        except FileNotFoundError:
            print(f"WARNING: {candidates_file} not found")
            return jsonify({
                "status": "error",
                "message": f"Candidates file not found: {candidates_file}"
            }), 404
        finally:
            # Clean up candidate file
            try:
                if os.path.exists(candidates_file):
                    os.remove(candidates_file)
                    print(f"Cleaned up {candidates_file}")
            except Exception as e:
                print(f"WARNING: Could not delete {candidates_file}: {e}")
        
        if len(candidates) == 0:
            return jsonify({
                "status": "success",
                "message": "No candidates to process",
                "duplicates": [],
                "total_candidates": 0
            })
        
        print("Loading dataset from CSV...")
        dataset_csv = f'/data/{dataset_name}.csv'
        all_records = pd.read_csv(dataset_csv, sep=csv_delimiter, header=0, on_bad_lines='skip')
        print(f"Loaded {len(all_records)} records")
        
        # Create a lookup dict for fast access
        records_dict = {str(row['id']): row for _, row in all_records.iterrows()}
        
        #  Load trained model and tokenizer
        print(f"Loading model from: {model_name}")
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"Model not found at {model_name}. Please train the model first.")
        if not os.path.exists(tokenizer_name):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_name}. Please train the model first.")
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Model loaded on device: {device}")
        
        print("Creating text pairs...")
        candidate_pairs = []
        
        missing_count = 0
        for _, row in candidates.iterrows():
            id1, id2 = str(row['id1']), str(row['id2'])
            
            # Get text for both records
            rec1 = records_dict.get(id1)
            rec2 = records_dict.get(id2)
            
            if rec1 is not None and rec2 is not None:
                text1 = f"{rec1.get('title', '')} {rec1.get('authors', '')}"
                text2 = f"{rec2.get('title', '')} {rec2.get('authors', '')}"
                candidate_pairs.append({
                    'id1': id1,
                    'id2': id2,
                    'text1': text1,
                    'text2': text2
                })
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Skipped {missing_count} pairs (missing records)")
        print(f"Processing {len(candidate_pairs)} pairs")
        
        if len(candidate_pairs) > 0 and len(candidate_pairs) <= 10:
            print("\nSample pairs:")
            for i, pair in enumerate(candidate_pairs[:5]):
                print(f"  [{pair['id1']}, {pair['id2']}]: {pair['text1'][:50]}... vs {pair['text2'][:50]}...")
        
        #  Batch inference
        duplicate_pairs = []
        total_pairs = len(candidate_pairs)
        
        for i in range(0, total_pairs, batch_size):
            batch = candidate_pairs[i:min(i + batch_size, total_pairs)]
            texts = [[pair['text1'], pair['text2']] for pair in batch]
            
            # Tokenize
            encoded = tokenizer.batch_encode_plus(
                texts,
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_masks = encoded['attention_mask'].to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                confidences = probs[:, 1].cpu().numpy()  # Probability of class 1 (duplicate)
            
            # Collect duplicates above threshold
            for j, (pred, conf) in enumerate(zip(predictions, confidences)):
                if pred == 1 and conf >= confidence_threshold:
                    pair = batch[j]
                    duplicate_pairs.append({
                        'id1': pair['id1'],
                        'id2': pair['id2'],
                        'confidence': float(conf)
                    })
        
        print(f"Found {len(duplicate_pairs)} duplicate pairs (threshold: {confidence_threshold})")
        
        # Format output
        if output_format == 'deduplicated':
            # Return deduplicated dataset (keep one representative from each duplicate group)
            # Build equivalence classes using union-find
            parent = {}
            
            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            # Group duplicates
            for pair in duplicate_pairs:
                union(pair['id1'], pair['id2'])
            
            # Get unique representatives from all_records
            unique_ids = set(all_records['id'].astype(str))
            groups = {}
            for id_str in unique_ids:
                root = find(id_str)
                if root not in groups:
                    groups[root] = []
                groups[root].append(id_str)
            
            # Keep one from each group (the root)
            deduplicated_ids = set(groups.keys())
            deduplicated_records = all_records[all_records['id'].astype(str).isin(deduplicated_ids)]
            
            print(f"Deduplicated: {len(all_records)} -> {len(deduplicated_records)} records")
            
            return jsonify({
                "status": "success",
                "message": f"Inference completed. Found {len(duplicate_pairs)} duplicate pairs.",
                "output_format": "deduplicated",
                "original_count": len(all_records),
                "deduplicated_count": len(deduplicated_records),
                "duplicate_pairs_found": len(duplicate_pairs),
                "deduplicated_data": deduplicated_records.to_dict(orient='records')
            })
        else:
            # Return duplicate pairs
            return jsonify({
                "status": "success",
                "message": f"Inference completed. Found {len(duplicate_pairs)} duplicate pairs.",
                "output_format": "pairs",
                "total_candidate_pairs": total_pairs,
                "duplicate_pairs": duplicate_pairs,
                "confidence_threshold": confidence_threshold
            })
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Inference error: {error_details}")
        return jsonify({"status": "error", "message": str(e), "details": error_details}), 500

@app.route('/inference_jaccard', methods=['POST'])
def inference_jaccard():
    try:
        print("Received Jaccard inference request.")
        data = request.get_json()
        
        query = data.get('query')
        dataset = data.get('dataset')
        dataset_name = data.get('dataset_name')
        csv_delimiter = data.get('csv_delimiter', '>')
        jaccard_threshold = data.get('jaccard_threshold', 0.5)
        output_format = data.get('output_format', 'pairs')
        
        request_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        candidates_file = f'/data/candidates_{request_id}.csv'
        print(f"Using unique candidates file: {candidates_file}")
        
        print(f"Running query: {query}")
        url = "http://api:8080/api/query"
        params = {"q": query, "page": 0, "offset": -1, "candidatesFile": candidates_file}
        response = requests.post(url, params=params, timeout=120)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            raise Exception(f"Java API query failed with status {response.status_code}")
        try:
            candidates = pd.read_csv(candidates_file)
            print(f"Loaded {len(candidates)} candidates")
        except FileNotFoundError:
            return jsonify({"status": "error", "message": f"Candidates file not found"}), 404
        finally:
            # Clean up candidate file
            try:
                if os.path.exists(candidates_file):
                    os.remove(candidates_file)
                    print(f"Cleaned up {candidates_file}")
            except Exception as e:
                print(f"WARNING: Could not delete {candidates_file}: {e}")
        
        if len(candidates) == 0:
            return jsonify({"status": "success", "message": "No candidates", "duplicates": [], "total_candidates": 0})
        
        print("Loading dataset from CSV...")
        dataset_csv = f'/data/{dataset_name}.csv'
        all_records = pd.read_csv(dataset_csv, sep=csv_delimiter, header=0, on_bad_lines='skip')
        print(f"Loaded {len(all_records)} records")
        
        records_dict = {str(row['id']): row for _, row in all_records.iterrows()}
        
        def jaccard_similarity(str1, str2):
            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            if len(set1) == 0 and len(set2) == 0:
                return 0.0
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0
        
        print("Computing Jaccard similarities...")
        duplicate_pairs = []
        missing_count = 0
        
        for _, row in candidates.iterrows():
            id1, id2 = str(row['id1']), str(row['id2'])
            rec1 = records_dict.get(id1)
            rec2 = records_dict.get(id2)
            
            if rec1 is not None and rec2 is not None:
                text1 = f"{rec1.get('title', '')} {rec1.get('authors', '')}"
                text2 = f"{rec2.get('title', '')} {rec2.get('authors', '')}"
                similarity = jaccard_similarity(text1, text2)
                
                if similarity >= jaccard_threshold:
                    duplicate_pairs.append({'id1': id1, 'id2': id2, 'confidence': float(similarity)})
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Skipped {missing_count} pairs (missing records)")
        print(f"Found {len(duplicate_pairs)} duplicates (threshold: {jaccard_threshold})")
        
        if output_format == 'deduplicated':
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            for pair in duplicate_pairs:
                union(pair['id1'], pair['id2'])
            
            unique_ids = set(all_records['id'].astype(str))
            groups = {}
            for id_str in unique_ids:
                root = find(id_str)
                if root not in groups:
                    groups[root] = []
                groups[root].append(id_str)
            
            deduplicated_ids = set(groups.keys())
            deduplicated_records = all_records[all_records['id'].astype(str).isin(deduplicated_ids)]
            print(f"Deduplicated: {len(all_records)} -> {len(deduplicated_records)} records")
            
            return jsonify({
                "status": "success",
                "message": f"Jaccard inference completed. Found {len(duplicate_pairs)} duplicate pairs.",
                "output_format": "deduplicated",
                "original_count": len(all_records),
                "deduplicated_count": len(deduplicated_records),
                "duplicate_pairs_found": len(duplicate_pairs),
                "deduplicated_data": deduplicated_records.to_dict(orient='records')
            })
        else:
            return jsonify({
                "status": "success",
                "message": f"Jaccard inference completed. Found {len(duplicate_pairs)} duplicate pairs.",
                "output_format": "pairs",
                "total_candidate_pairs": len(candidates),
                "duplicate_pairs": duplicate_pairs,
                "jaccard_threshold": jaccard_threshold
            })
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Jaccard inference error: {error_details}")
        return jsonify({"status": "error", "message": str(e), "details": error_details}), 500
    
@app.route('/inference_jaccard_clean_clean', methods=['POST'])
def inference_jaccard_clean_clean():
    try:
        print("Received Jaccard clean-clean inference request.")
        data = request.get_json()

        query = data.get('query')
        dataset1_name = data.get('dataset1_name')
        dataset2_name = data.get('dataset2_name')
        csv_delimiter1 = data.get('csv_delimiter1', '>')
        csv_delimiter2 = data.get('csv_delimiter2', '>')
        jaccard_threshold = data.get('jaccard_threshold', 0.5)
        output_format = data.get('output_format', 'pairs')

        request_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        candidates_file = f'/data/candidates_{request_id}.csv'
        print(f"Using unique candidates file: {candidates_file}")

        print(f"Running query: {query}")
        url = "http://api:8080/api/query"
        params = {"q": query, "page": 0, "offset": -1, "candidatesFile": candidates_file}
        response = requests.post(url, params=params, timeout=120)
        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"Java API query failed with status {response.status_code}")
        try:
            candidates = pd.read_csv(candidates_file)
            print(f"Loaded {len(candidates)} candidates")
        except FileNotFoundError:
            return jsonify({"status": "error", "message": f"Candidates file not found"}), 404
        finally:
            # Clean up candidate file
            try:
                if os.path.exists(candidates_file):
                    os.remove(candidates_file)
                    print(f"Cleaned up {candidates_file}")
            except Exception as e:
                print(f"WARNING: Could not delete {candidates_file}: {e}")

        if len(candidates) == 0:
            return jsonify({"status": "success", "message": "No candidates", "duplicates": [], "total_candidates": 0})

        print("Loading dataset1 from CSV...")
        dataset1_csv = f'/data/{dataset1_name}.csv'
        all_records1 = pd.read_csv(dataset1_csv, sep=csv_delimiter1, header=0, on_bad_lines='skip')
        print(f"Loaded {len(all_records1)} records from dataset1")

        print("Loading dataset2 from CSV...")
        dataset2_csv = f'/data/{dataset2_name}.csv'
        all_records2 = pd.read_csv(dataset2_csv, sep=csv_delimiter2, header=0, on_bad_lines='skip')
        print(f"Loaded {len(all_records2)} records from dataset2")

        records_dict1 = {str(row['id']): row for _, row in all_records1.iterrows()}
        records_dict2 = {str(row['id']): row for _, row in all_records2.iterrows()}

        def jaccard_similarity(str1, str2):
            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            if len(set1) == 0 and len(set2) == 0:
                return 0.0
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0

        print("Computing Jaccard similarities...")
        duplicate_pairs = []
        missing_count = 0

        for _, row in candidates.iterrows():
            id1, id2 = str(row['id1']), str(row['id2'])
            rec1 = records_dict1.get(id1)
            rec2 = records_dict2.get(id2)

            if rec1 is not None and rec2 is not None:
                text1 = f"{rec1.get('title', '')} {rec1.get('authors', '')}"
                text2 = f"{rec2.get('title', '')} {rec2.get('authors', '')}"
                similarity = jaccard_similarity(text1, text2)

                if similarity >= jaccard_threshold:
                    duplicate_pairs.append({'id1': id1, 'id2': id2, 'confidence': float(similarity)})
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"Skipped {missing_count} pairs (missing records)")
        print(f"Found {len(duplicate_pairs)} duplicates (threshold: {jaccard_threshold})")

        if output_format == 'deduplicated':
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for pair in duplicate_pairs:
                union(pair['id1'], pair['id2'])

            # For clean-clean, deduplication is across both datasets
            unique_ids = set([str(x) for x in all_records1['id']]) | set([str(x) for x in all_records2['id']])
            groups = {}
            for id_str in unique_ids:
                root = find(id_str)
                if root not in groups:
                    groups[root] = []
                groups[root].append(id_str)

            # For reporting, keep one record from each group, prefer dataset1 if present
            deduplicated_ids = set(groups.keys())
            deduplicated_records = []
            for root_id in deduplicated_ids:
                # Try to get from dataset1, else dataset2
                rec = records_dict1.get(root_id) or records_dict2.get(root_id)
                if rec is not None:
                    deduplicated_records.append(rec)

            print(f"Deduplicated: {len(all_records1) + len(all_records2)} -> {len(deduplicated_records)} records")

            return jsonify({
                "status": "success",
                "message": f"Jaccard clean-clean inference completed. Found {len(duplicate_pairs)} duplicate pairs.",
                "output_format": "deduplicated",
                "original_count": len(all_records1) + len(all_records2),
                "deduplicated_count": len(deduplicated_records),
                "duplicate_pairs_found": len(duplicate_pairs),
                "deduplicated_data": [dict(r) for r in deduplicated_records]
            })
        else:
            return jsonify({
                "status": "success",
                "message": f"Jaccard clean-clean inference completed. Found {len(duplicate_pairs)} duplicate pairs.",
                "output_format": "pairs",
                "total_candidate_pairs": len(candidates),
                "duplicate_pairs": duplicate_pairs,
                "jaccard_threshold": jaccard_threshold
            })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Jaccard clean-clean inference error: {error_details}")
        return jsonify({"status": "error", "message": str(e), "details": error_details}), 500

def cleanup_old_files_background():
    """Background thread that periodically cleans up old temporary files"""
    while True:
        try:
            time.sleep(3600)  # Run every hour
            current_time = time.time()
            
            for filepath, info in list(temporary_files.items()):
                age_hours = (current_time - info['created_at']) / 3600
                
                if age_hours >= info['cleanup_hours']:
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            print(f"[Background Cleanup] Removed old file: {filepath} (age: {age_hours:.2f} hours)")
                        del temporary_files[filepath]
                    except Exception as e:
                        print(f"[Background Cleanup] Error removing {filepath}: {e}")
        except Exception as e:
            print(f"[Background Cleanup] Error in cleanup thread: {e}")
    
if __name__ == '__main__':
    # Start background cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files_background, daemon=True)
    cleanup_thread.start()
    print("Started background cleanup thread")
    
    from waitress import serve
    print("Starting production server on port 5000...")
    serve(app, host='0.0.0.0', port=5000, threads=4)

