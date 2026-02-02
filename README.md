## Usage

### Start Docker Services

```bash
docker-compose up -d
```

This starts three services:
- **api** (port 8080) - Java query engine
- **bert** (port 5000) - Python Flask API for model training and inference
- **llm** - Python Arrow Flight server for BERT processing

### Upload Dataset

```powershell
$uploadPayload = @{
    filename = "publications.csv"
    data = "id,title,authors,venue,year`n1,Paper Title,Author Name,Conference,2023"
    temporary = $true
    cleanup_hours = 12
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/upload-dataset" -Method POST -ContentType "application/json" -Body $uploadPayload
```

Or upload ground truth:

```powershell
$groundTruthPayload = @{
    filename = "ground_truth_publications.csv"
    data = "id1;id2`n1;2`n3;4"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/upload-dataset" -Method POST -ContentType "application/json" -Body $groundTruthPayload
```

### Train Model

```powershell
$trainPayload = @{
    dataset_name = "publications"
    query = "SELECT DEDUP * FROM publications WHERE MOD(id, 1000) = 0"
    ground_truth = "/data/ground_truth_publications.csv"
    ground_truth_delimiter = ";"
    csv_delimiter = ">"
    model = "prajjwal1/bert-tiny"
    epochs = 1
    batch_size = 128
    learning_rate = 0.00005
    max_seq_length = 128
    evaluation_metric = "f1_score"
    class_weights = @(1.0, 5.0)
    model_name = "/app/models/my_distilbert_model"
    tokenizer_name = "/app/models/my_distilbert_tokenizer"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/train" -Method POST -ContentType "application/json" -Body $trainPayload -TimeoutSec 600
```

### Run Inference (BERT)

```powershell
$inferencePayload = @{
    dataset_name = "publications"
    query = "SELECT DEDUP * FROM publications WHERE MOD(id, 1000) = 0"
    csv_delimiter = ">"
    model_name = "/app/models/my_distilbert_model"
    tokenizer_name = "/app/models/my_distilbert_tokenizer"
    batch_size = 128
    max_seq_length = 128
    confidence_threshold = 0.5
    output_format = "pairs"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/inference" -Method POST -ContentType "application/json" -Body $inferencePayload -TimeoutSec 300
```

### Run Inference (Jaccard - No Training Required)

```powershell
$jaccardPayload = @{
    dataset_name = "publications"
    query = "SELECT DEDUP * FROM publications WHERE MOD(id, 1000) = 0"
    csv_delimiter = ">"
    jaccard_threshold = 0.5
    output_format = "pairs"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/inference_jaccard" -Method POST -ContentType "application/json" -Body $jaccardPayload -TimeoutSec 300
```

### API Documentation

Full API documentation is available in `QueryMLER_-_API_Specifications.yaml` (OpenAPI 3.0 format).
