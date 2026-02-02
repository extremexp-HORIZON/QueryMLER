package org.imsi.queryERAPI.dto;

/**
 * DTO for dataset loading requests
 */
public class DatasetLoadRequest {
    private String datasetPath;
    private String datasetName;
    private String schemaName;
    private boolean temporary;

    public DatasetLoadRequest() {
    }

    public DatasetLoadRequest(String datasetPath, String datasetName, String schemaName, boolean temporary) {
        this.datasetPath = datasetPath;
        this.datasetName = datasetName;
        this.schemaName = schemaName;
        this.temporary = temporary;
    }

    public String getDatasetPath() {
        return datasetPath;
    }

    public void setDatasetPath(String datasetPath) {
        this.datasetPath = datasetPath;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public void setDatasetName(String datasetName) {
        this.datasetName = datasetName;
    }

    public String getSchemaName() {
        return schemaName;
    }

    public void setSchemaName(String schemaName) {
        this.schemaName = schemaName;
    }

    public boolean isTemporary() {
        return temporary;
    }

    public void setTemporary(boolean temporary) {
        this.temporary = temporary;
    }
}
