package org.imsi.queryERAPI.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.imsi.queryERAPI.util.PagedResult;
import org.imsi.queryERAPI.util.ResultSetToJsonMapper;
import org.imsi.queryEREngine.imsi.er.QueryEngine;
import org.imsi.queryEREngine.imsi.er.BigVizUtilities.BigVizOutput;
import org.imsi.queryEREngine.imsi.er.Utilities.DumpDirectories;
import org.imsi.queryEREngine.imsi.er.Utilities.SerializationUtilities;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.io.FileWriter;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import javax.sql.rowset.CachedRowSet;
import javax.sql.rowset.RowSetFactory;
import javax.sql.rowset.RowSetProvider;

import static org.springframework.http.ResponseEntity.ok;


@RestController()
@RequestMapping("/api")
@CrossOrigin
public class QueryController {

	ResultSet rs;
	CachedRowSet rowset;
	List<ObjectNode> results = null;
	DumpDirectories dumpDirectories = new DumpDirectories();
	String query = "";
	@PostMapping("/query")
	public ResponseEntity<String> query(@RequestParam(value = "q", required = true) String q,
			@RequestParam(value = "page", required = false) Integer page, 
			@RequestParam(value = "offset", required = false) Integer offset,
			@RequestParam(value = "candidatesFile", required = false) String candidatesFile) throws JsonProcessingException, SQLException  {

		return queryResult(q, page, offset, candidatesFile);

	}
	
	@PostMapping("/query-inference")
	public ResponseEntity<String> queryInference(@RequestParam(value = "q", required = true) String q,
											  @RequestParam(value = "page", required = false) int page,
											  @RequestParam(value = "offset", required = false) int offset,
											  @RequestParam(value = "model_name", required = false) String modelName,
											  @RequestParam(value = "tokenizer_name", required = false) String tokenizerName,
                                                  @RequestParam(value = "inference_type", required = false) String inferenceType,
											  @RequestParam(value = "batch_size", required = false, defaultValue = "32") int batchSize,
											  @RequestParam(value = "candidatesFile", required = false) String candidatesFile) throws JsonProcessingException, SQLException {
        
        QueryParametersInference params = new QueryParametersInference(q, page, offset, modelName, tokenizerName, batchSize);
        writeParamsToJson(params);
        return queryResult(q, page, offset, candidatesFile);
	}	private void writeParamsToJson(QueryParametersInference params) {
        try {
			final ObjectMapper objectMapper = new ObjectMapper();

            String json = objectMapper.writeValueAsString(params);
            File jsonFile = new File("query_params.json");

			try (FileWriter fileWriter = new FileWriter(jsonFile)) {
				fileWriter.write(json);
				System.out.println("JSON file written to: " + jsonFile.getAbsolutePath());
			}	
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
	@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
    static class QueryParametersInference {
		private String q;
		private int page;
		private int offset;
		private String modelName;
		private String tokenizerName;
		private int batchSize;
		private final String training;

        public QueryParametersInference(String q, int page, int offset, String modelName, String tokenizerName, int batchSize) {
            this.q = q;
            this.page = page;
            this.offset = offset;
            this.modelName = modelName;
            this.tokenizerName = tokenizerName;
            this.batchSize = batchSize;
			this.training = "false";
        }
    }

    @PostMapping("/query-train")
    public ResponseEntity<String> trainQueryEndpoint(
            @RequestParam(value = "q", required = true) String q,
			@RequestParam(value = "page", required = false) int page,
			@RequestParam(value = "offset", required = false) int offset,
			@RequestParam(value = "dataset", required = false) String dataset,
            @RequestParam(value = "ground_truth", required = false) String groundTruth,
            @RequestParam(value = "dataset_name", required = false) String datasetName,
            @RequestParam(value = "train_df_name", required = false) String trainDfName,
            @RequestParam(value = "train_csv", required = false) String trainCsv,
            @RequestParam(value = "model", required = false) String model,
            @RequestParam(value = "epochs", required = false, defaultValue = "1") int epochs,
            @RequestParam(value = "batch_size", required = false, defaultValue = "32") int batchSize,
            @RequestParam(value = "learning_rate", required = false, defaultValue = "0.001") double learningRate,
            @RequestParam(value = "max_seq_length", required = false, defaultValue = "128") int maxSeqLength,
            @RequestParam(value = "evaluation_metric", required = false, defaultValue = "accuracy") String evaluationMetric,
            @RequestParam(value = "confidence_threshold", required = false, defaultValue = "0.5") double confidenceThreshold,
            @RequestParam(value = "top_k_predictions", required = false, defaultValue = "3") int topKPredictions,
            @RequestParam(value = "model_name", required = false) String modelName,
            @RequestParam(value = "tokenizer_name", required = false) String tokenizerName,
            @RequestParam(value = "class_weights", required = false) double[] classWeights,
            @RequestParam(value = "loss_func_type", required = false, defaultValue = "CrossEntropyLoss") String lossFuncType
    ) throws JsonProcessingException, SQLException {
        QueryParametersTrain params = new QueryParametersTrain(
                dataset, groundTruth, q, datasetName, trainDfName, trainCsv, model,
                epochs, batchSize, learningRate, maxSeqLength, evaluationMetric, confidenceThreshold,
                topKPredictions, modelName, tokenizerName, classWeights, lossFuncType
        );
        saveTrainParamsToJson(params);
        return queryResult(q, page, offset, null);
    }

    private void saveTrainParamsToJson(QueryParametersTrain params) {
        try {
            final ObjectMapper objectMapper = new ObjectMapper();
            String json = objectMapper.writeValueAsString(params);
            File jsonFile = new File("query_params.json");

            try (FileWriter fileWriter = new FileWriter(jsonFile)) {
                fileWriter.write(json);
                System.out.println("JSON file written to: " + jsonFile.getAbsolutePath());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
    static class QueryParametersTrain {
        private String dataset;
        private String groundTruth;
        private String query;
        private String datasetName;
        private String trainDfName;
        private String trainCsv;
        private String model;
        private int epochs;
        private int batchSize;
        private double learningRate;
        private int maxSeqLength;
        private String evaluationMetric;
        private double confidenceThreshold;
        private int topKPredictions;
        private String modelName;
        private String tokenizerName;
        private double[] classWeights;
        private String lossFuncType;
		private final String training;

        public QueryParametersTrain(String dataset, String groundTruth, String query, String datasetName, String trainDfName, String trainCsv,
                                    String model, int epochs, int batchSize, double learningRate, int maxSeqLength, String evaluationMetric,
                                    double confidenceThreshold, int topKPredictions, String modelName, String tokenizerName,
                                    double[] classWeights, String lossFuncType) {
            this.dataset = dataset;
            this.groundTruth = groundTruth;
            this.query = query;
            this.datasetName = datasetName;
            this.trainDfName = trainDfName;
            this.trainCsv = trainCsv;
            this.model = model;
            this.epochs = epochs;
            this.batchSize = batchSize;
            this.learningRate = learningRate;
            this.maxSeqLength = maxSeqLength;
            this.evaluationMetric = evaluationMetric;
            this.confidenceThreshold = confidenceThreshold;
            this.topKPredictions = topKPredictions;
            this.modelName = modelName;
            this.tokenizerName = tokenizerName;
            this.classWeights = classWeights;
            this.lossFuncType = lossFuncType;
			this.training = "true";
        }
    }

	@PostMapping("/query-rv")
	public ResponseEntity<String> query(@RequestParam(value = "q", required = true) String q) throws JsonProcessingException, SQLException  {

		return liResult(q);
		
	}

	@PostMapping("/columns")
	public ResponseEntity<String> columns(@RequestParam(value = "d", required = true) String dataset) throws JsonProcessingException, SQLException  {
		String q = "SELECT * FROM " + dataset + " LIMIT 3";
		QueryEngine qe = new QueryEngine();
		ObjectMapper mapper = new ObjectMapper();
		rs = qe.runQuery(q);
		ResultSetMetaData rsmd = rs.getMetaData();
		List<String> columns = new ArrayList<>();
		for(int i = 0; i < rsmd.getColumnCount(); i++) {
			columns.add(rsmd.getColumnName(i+1));
		}
		return ok(mapper.writeValueAsString(columns));

	}
	public ResponseEntity<String> liResult(String q) throws SQLException, JsonProcessingException {

		ObjectMapper mapper = new ObjectMapper();
		QueryEngine qe = new QueryEngine();
		if(!this.query.contentEquals(q)) {
			rs = qe.runQuery(q);		
			if(rs != null) {
				BigVizOutput bigVizOutput = (BigVizOutput) SerializationUtilities.loadSerializedObject(dumpDirectories.getLiFilePath());
				return ok(mapper.writeValueAsString(bigVizOutput));
			}
			
		}

		
		return null;
	}
	
	public ResponseEntity<String> queryResult(String q, Integer page, Integer offset, String candidatesFile) throws SQLException, JsonProcessingException {
		page +=1;

		ObjectMapper mapper = new ObjectMapper();
		QueryEngine qe = new QueryEngine();

		// Set candidates file path as system property if provided
		if (candidatesFile != null && !candidatesFile.isEmpty()) {
			System.setProperty("queryER.candidatesFile", candidatesFile);
			System.out.println("Set candidatesFile system property: " + candidatesFile);
		} else {
			// Use default if not provided
			System.setProperty("queryER.candidatesFile", "/data/candidates.csv");
		}

		// Don't cache DEDUP queries - they need to run fresh each time
		boolean isDedupQuery = q.toUpperCase().contains("DEDUP");
		
		if(!this.query.contentEquals(q) || isDedupQuery) {
			rs = qe.runQuery(q);		
			RowSetFactory factory = RowSetProvider.newFactory();
			rowset = factory.createCachedRowSet();			 
			rowset.populate(rs);
			this.query = q;
		} else {
			// Reset cursor for cached non-DEDUP queries
			try {
				rowset.beforeFirst();
			} catch (SQLException e) {
				// Ignore if beforeFirst fails
			}
		}

		int end = rowset.size();
		int pages = (int) Math.floor(end / offset) + 1;
		
		int resultOffset = offset * page;
		int startOffset = resultOffset - offset;
		if(page == pages) {
			startOffset = offset * (page - 1);
			resultOffset = end;
			
		}
		if(resultOffset < offset || offset == -1) {
			startOffset = 1;
			resultOffset = end;
		}
		if(startOffset == 0) startOffset = 1;
		results = ResultSetToJsonMapper.mapCRS(rowset, startOffset, resultOffset);

		return ok(mapper.writeValueAsString(new PagedResult(pages, results, end)));
	}
	
	

}
