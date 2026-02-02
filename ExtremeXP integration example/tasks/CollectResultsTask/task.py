import os, sys
import eexp_engine.executionware.proactive_helper as ph
[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]

import json
from datetime import datetime

print("Running CollectResultsTask - Aggregating pipeline results")

# Load inference results from variables
inference_results_json = variables.get("InferenceResults")
print(f"Inference results loaded")

try:
    inference_data = json.loads(inference_results_json)
    print("Collecting final metrics...")

    duplicate_pairs_list = inference_data.get("duplicate_pairs", [])
    duplicate_pairs = len(duplicate_pairs_list)
    inference_status = inference_data.get("status", "unknown")

    summary = {
        "pipeline_status": "COMPLETED",
        "completion_time": datetime.now().isoformat(),
        "inference_results": {
            "duplicate_pairs_found": duplicate_pairs,
            "inference_status": inference_status
        },
        "inference_details": inference_data
    }

    print(f"\n{'='*50}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*50}")
    print(f"Status: {summary['pipeline_status']}")
    print(f"Completion Time: {summary['completion_time']}")
    print(f"Duplicate Pairs Found: {duplicate_pairs}")
    print(f"{'='*50}\n")

    resultMap.put("pipeline_status", summary['pipeline_status'])
    resultMap.put("duplicate_pairs_found", int(duplicate_pairs))
    resultMap.put("pipeline_completion_time", summary['completion_time'])

    # Store final results as JSON string
    variables.put("FinalResults", json.dumps(summary, indent=2))

    print("Results collection completed successfully!")
    
except Exception as e:
    print(f"Error collecting results: {e}")
    resultMap.put("pipeline_status", f"ERROR: {str(e)}")
    raise
