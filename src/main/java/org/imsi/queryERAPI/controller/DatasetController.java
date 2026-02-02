package org.imsi.queryERAPI.controller;

import org.imsi.queryERAPI.dto.DatasetLoadRequest;
import org.imsi.queryERAPI.service.DatasetService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * REST controller for dynamic dataset loading
 */
@RestController
@RequestMapping("/api")
public class DatasetController {

    @Autowired
    private DatasetService datasetService;

    /**
     * Load a new dataset dynamically and create its block index
     * 
     * POST /api/load-dataset
     * Body: {
     *   "datasetPath": "/data/my_dataset.csv",
     *   "datasetName": "my_dataset",
     *   "schemaName": "csv",
     *   "temporary": true
     * }
     */
    @PostMapping("/load-dataset")
    public ResponseEntity<Map<String, Object>> loadDataset(@RequestBody DatasetLoadRequest request) {
        Map<String, Object> response = new HashMap<>();
        
        try {
            // Validate request
            if (request.getDatasetPath() == null || request.getDatasetPath().isEmpty()) {
                response.put("status", "error");
                response.put("message", "datasetPath is required");
                return ResponseEntity.badRequest().body(response);
            }
            
            if (request.getDatasetName() == null || request.getDatasetName().isEmpty()) {
                response.put("status", "error");
                response.put("message", "datasetName is required");
                return ResponseEntity.badRequest().body(response);
            }
            
            // Load dataset
            String result = datasetService.loadDataset(
                request.getDatasetPath(),
                request.getDatasetName(),
                request.getSchemaName() != null ? request.getSchemaName() : "csv",
                request.isTemporary()
            );
            
            response.put("status", "success");
            response.put("message", result);
            response.put("datasetName", request.getDatasetName());
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            e.printStackTrace();
            response.put("status", "error");
            response.put("message", "Failed to load dataset: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }

    /**
     * Remove a dataset and its files
     * 
     * DELETE /api/dataset/{datasetName}
     */
    @DeleteMapping("/dataset/{datasetName}")
    public ResponseEntity<Map<String, Object>> removeDataset(@PathVariable String datasetName) {
        Map<String, Object> response = new HashMap<>();
        
        try {
            datasetService.removeDataset(datasetName);
            response.put("status", "success");
            response.put("message", "Dataset removed successfully");
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            e.printStackTrace();
            response.put("status", "error");
            response.put("message", "Failed to remove dataset: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }

    /**
     * Get all temporary datasets
     * 
     * GET /api/datasets/temporary
     */
    @GetMapping("/datasets/temporary")
    public ResponseEntity<Map<String, Object>> getTemporaryDatasets() {
        Map<String, Object> response = new HashMap<>();
        
        try {
            response.put("status", "success");
            response.put("datasets", datasetService.getTemporaryDatasets());
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            e.printStackTrace();
            response.put("status", "error");
            response.put("message", "Failed to get temporary datasets: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
}
