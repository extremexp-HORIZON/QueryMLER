package org.imsi.queryERAPI.scheduler;

import org.imsi.queryERAPI.service.DatasetService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.IOException;

/**
 * Scheduled task to clean up old temporary datasets
 */
@Component
public class CleanupScheduler {

    @Autowired
    private DatasetService datasetService;

    /**
     * Clean up temporary datasets older than 12 hours
     * Runs every hour
     */
    @Scheduled(fixedRate = 3600000) // 1 hour in milliseconds
    public void cleanupOldDatasets() {
        System.out.println("Running scheduled cleanup of temporary datasets...");
        
        try {
            int removed = datasetService.cleanupOldDatasets(12); // 12 hours
            
            if (removed > 0) {
                System.out.println("Cleanup complete: removed " + removed + " old temporary dataset(s)");
            } else {
                System.out.println("Cleanup complete: no old datasets to remove");
            }
            
        } catch (IOException e) {
            System.err.println("Error during cleanup: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Alternative: Clean up datasets older than 6 hours
     * Uncomment if you want more aggressive cleanup
     */
    // @Scheduled(fixedRate = 1800000) // 30 minutes
    // public void cleanupVeryOldDatasets() {
    //     try {
    //         datasetService.cleanupOldDatasets(6);
    //     } catch (IOException e) {
    //         System.err.println("Error during aggressive cleanup: " + e.getMessage());
    //     }
    // }
}
