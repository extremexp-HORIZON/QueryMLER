package org.imsi.queryERAPI.service;

import org.imsi.queryEREngine.apache.calcite.jdbc.CalciteConnection;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.schema.Table;
import org.imsi.queryEREngine.apache.calcite.util.Source;
import org.imsi.queryEREngine.apache.calcite.util.Sources;
import org.imsi.queryEREngine.imsi.calcite.adapter.enumerable.csv.CsvEnumerator;
import org.imsi.queryEREngine.imsi.calcite.adapter.enumerable.csv.CsvSchema;
import org.imsi.queryEREngine.imsi.calcite.adapter.enumerable.csv.CsvTranslatableTable;
import org.imsi.queryEREngine.imsi.er.BlockIndex.BaseBlockIndex;
import org.imsi.queryEREngine.imsi.er.BlockIndex.BlockIndexStatistic;
import org.imsi.queryEREngine.imsi.er.ConnectionPool.CalciteConnectionPool;
import org.imsi.queryEREngine.imsi.er.Utilities.DumpDirectories;
import org.imsi.queryEREngine.imsi.er.Utilities.SerializationUtilities;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.google.common.collect.ImmutableMap;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Service for dynamic dataset loading and block index creation
 */
@Service
public class DatasetService {

    @Value("${dump.path:data}")
    private String dumpPath;

    private DumpDirectories dumpDirectories;
    
    // Track temporary datasets with their creation times
    private Map<String, LocalDateTime> temporaryDatasets = new HashMap<>();

    @PostConstruct
    public void init() throws IOException {
        this.dumpDirectories = new DumpDirectories(dumpPath);
        this.dumpDirectories.generateDumpDirectories();
    }

    /**
     * Load a new dataset and create its block index
     * 
     * @param datasetPath Path to the CSV file
     * @param datasetName Name of the dataset/table
     * @param schemaName Schema name
     * @param temporary Whether this is a temporary dataset
     * @return Success message
     * @throws IOException
     * @throws SQLException
     */
    public String loadDataset(String datasetPath, String datasetName, String schemaName, boolean temporary) 
            throws IOException, SQLException {
        
        System.out.println("Loading dataset: " + datasetName + " from path: " + datasetPath);
        
        File datasetFile = new File(datasetPath);
        if (!datasetFile.exists()) {
            throw new IOException("Dataset file not found: " + datasetPath);
        }

        Source source = Sources.of(datasetFile);
        
        // Create the CSV table
        CsvTranslatableTable table = new CsvTranslatableTable(source, datasetName, null);
        
        // Determine the key field (id or rec_id)
        List<RelDataTypeField> fields = table.getRowType(new org.imsi.queryEREngine.apache.calcite.jdbc.JavaTypeFactoryImpl()).getFieldList();
        List<String> fieldNames = new ArrayList<String>();
        for(RelDataTypeField field : fields) {
            fieldNames.add(field.getName());
        }
        
        String[] keys = {"rec_id", "id"};
        for(String key : keys) {
            if(fieldNames.contains(key)) {
                table.setKey(fieldNames.indexOf(key));
                break;
            }
        }
        
        // Create block index
        BaseBlockIndex blockIndex = createBlockIndex(table, datasetName);
        
        // Force schema to rescan by clearing the tableMap
        // The next query will trigger getTableMap() which will recreate the map
        // including all CSV files in /data directory (including newly uploaded ones)
        CsvSchema.tableMap = null;
        
        System.out.println("Cleared tableMap to force schema rescan");
        
        // Track if temporary
        if (temporary) {
            temporaryDatasets.put(datasetName, LocalDateTime.now());
            System.out.println("Marked dataset as temporary: " + datasetName);
        }
        
        System.out.println("Dataset loaded successfully: " + datasetName);
        return "Dataset " + datasetName + " loaded successfully with block index";
    }

    /**
     * Create block index for a table
     */
    private BaseBlockIndex createBlockIndex(CsvTranslatableTable table, String tableName) throws IOException {
        BaseBlockIndex blockIndex = new BaseBlockIndex();
        
        System.out.println("Creating Block Index for: " + tableName);
        double start = System.currentTimeMillis();
        
        AtomicBoolean ab = new AtomicBoolean();
        ab.set(false);
        HashMap<Integer, Long> offsetIndex = new HashMap<>();
        
        @SuppressWarnings({ "unchecked", "rawtypes" })
        CsvEnumerator<Object[]> enumerator = new CsvEnumerator(table.getSource(), ab,
                table.getFieldTypes(), table.getKey(), offsetIndex);

        int tableSize = blockIndex.createBlockIndex(enumerator, table.getKey());
        blockIndex.buildBlocks();
        
        double end = System.currentTimeMillis();
        System.out.println("Block Index created in: " + (end - start)/1000 + " seconds");
        
        blockIndex.sortIndex();
        blockIndex.storeBlockIndex(dumpDirectories.getBlockIndexDirPath(), tableName);
        SerializationUtilities.storeSerializedObject(offsetIndex, dumpDirectories.getOffsetsDirPath() + tableName);
        
        // Create and store statistics
        BlockIndexStatistic blockIndexStatistic = new BlockIndexStatistic(
                blockIndex.getInvertedIndex(),
                blockIndex.getEntitiesToBlocks(), 
                tableName);
        blockIndexStatistic.setTableSize(tableSize);
        blockIndex.setBlockIndexStatistic(blockIndexStatistic);
        blockIndexStatistic.storeStatistics();
        
        System.out.println("Block Index stored successfully!");
        return blockIndex;
    }

    /**
     * Get all temporary datasets with their ages
     */
    public Map<String, LocalDateTime> getTemporaryDatasets() {
        return new HashMap<>(temporaryDatasets);
    }

    /**
     * Remove a dataset and its associated files
     */
    public void removeDataset(String datasetName) throws IOException {
        System.out.println("Removing dataset: " + datasetName);
        
        // Remove from table map
        if (CsvSchema.tableMap != null) {
            CsvSchema.tableMap.remove(datasetName);
            CsvSchema.tableMap.remove(dumpDirectories.getBlockIndexDirPath() + datasetName + "InvertedIndex");
        }
        
        // Delete block index files
        File invertedIndex = new File(dumpDirectories.getBlockIndexDirPath() + datasetName + "InvertedIndex");
        if (invertedIndex.exists()) {
            invertedIndex.delete();
        }
        
        File entitiesToBlocks = new File(dumpDirectories.getBlockIndexDirPath() + datasetName + "EntitiesToBlocks");
        if (entitiesToBlocks.exists()) {
            entitiesToBlocks.delete();
        }
        
        // Delete offset files
        File offsetFile = new File(dumpDirectories.getOffsetsDirPath() + datasetName);
        if (offsetFile.exists()) {
            offsetFile.delete();
        }
        
        // Delete statistics
        File statsFile = new File(dumpDirectories.getBlockIndexStatsDirPath() + datasetName + ".json");
        if (statsFile.exists()) {
            statsFile.delete();
        }
        
        // Remove from temporary tracking
        temporaryDatasets.remove(datasetName);
        
        System.out.println("Dataset removed: " + datasetName);
    }

    /**
     * Clean up datasets older than specified hours
     */
    public int cleanupOldDatasets(int hoursOld) throws IOException {
        LocalDateTime cutoffTime = LocalDateTime.now().minusHours(hoursOld);
        List<String> datasetsToRemove = new ArrayList<>();
        
        for (Map.Entry<String, LocalDateTime> entry : temporaryDatasets.entrySet()) {
            if (entry.getValue().isBefore(cutoffTime)) {
                datasetsToRemove.add(entry.getKey());
            }
        }
        
        for (String datasetName : datasetsToRemove) {
            try {
                removeDataset(datasetName);
                System.out.println("Cleaned up old dataset: " + datasetName);
            } catch (Exception e) {
                System.err.println("Error removing dataset " + datasetName + ": " + e.getMessage());
            }
        }
        
        return datasetsToRemove.size();
    }
}
