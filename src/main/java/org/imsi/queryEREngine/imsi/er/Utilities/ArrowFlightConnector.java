package org.imsi.queryEREngine.imsi.er.Utilities;

import org.apache.arrow.flight.*;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.types.pojo.ArrowType;

import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;

/**
 * Class responsible for initiating a connection to an existing arrow flight server on a given port.
 * You can then initiate server calls using the public functions of this class, such as storing data,
 * retrieving data, executing actions, and awaiting responses.
 */
public class ArrowFlightConnector {
    private static final BufferAllocator ALLOCATOR = new RootAllocator(Long.MAX_VALUE);
    private static FlightClient client;
    private final Location location;
    private static final int FLIGHT_TIMEOUT = 2;

    /**
     * Initiates a remote connection at an Arrow Flight server on the given port
     * The data can later be retrieved or deleted using the same descriptor
     * <p>
     * @param port The port at which the connection will be established
     */
    public ArrowFlightConnector(int port) {
        String host = System.getenv().getOrDefault("FLIGHT_SERVER_HOST", "llm");
        this.location = Location.forGrpcInsecure(host, port);
        System.out.println("Attempting to connect to server at " + host + ":" + port + "...");
        client = FlightClient.builder(ALLOCATOR, this.location).build();
        System.out.println("Connection established");
    }

    /**
     * Asks the server to store data under "descriptor" identifier
     * The data can later be retrieved or deleted using the same descriptor
     * <p>
     * @param vsr A VectorSchemaRoot table that holds data to be stored
     * @param descriptor A unique identifier under which the data is stored
     * @return VectorSchemaRoot table with all the resulting data
     */
    public void putData(VectorSchemaRoot vsr, String descriptor){
        Schema schema = vsr.getSchema();
    
        // Print the schema to inspect it
        System.out.println("Schema: " + schema.toString());
        // for (Field field : schema.getFields()) {
        //     System.out.println("Field: " + field.getName() + ", Type: " + field.getFieldType());
            
        //     if (field.getFieldType().getType() instanceof ArrowType.List) {
        //         // If the field is a list, find its child field by name
        //         ArrowType.List listType = (ArrowType.List) field.getFieldType().getType();
        //         String childFieldName = listType.getField().getName();
        //         System.out.println("Child Field Name of " + field.getName() + ": " + childFieldName);
        //     }
        // }
    
        FlightClient.ClientStreamListener listener = client.startPut(
                FlightDescriptor.path(descriptor),
                vsr, new AsyncPutListener());
        listener.putNext();
        listener.completed();
        listener.getResult();
    }

    /**
     * Asks the server for data under "descriptor" identifier
     * <p>
     * @param descriptor A unique identifier under which the data is stored
     * @return VectorSchemaRoot table with all the resulting data
     */
    public VectorSchemaRoot getData(String descriptor){
        FlightStream flightStream = client.getStream(new Ticket(
                FlightDescriptor.path(descriptor).getPath()
                        .get(0).getBytes(StandardCharsets.UTF_8)));
        int batch = 0;
        VectorSchemaRoot vectorSchemaRootReceived = flightStream.getRoot();
        while (flightStream.next()) {
            batch++;
        }
        return vectorSchemaRootReceived;

    }

    /**
     * Asks the server if the prediction results are ready.
     * Server response is handled and true or false is returned accordingly.
     * Program will stall until server responds or a timeout occurs
     * <p>
     * @return boolean
     */
    public boolean isPredictionReady(){
        Iterator<Result> actionResult = client.doAction(new Action("is_ready",
                FlightDescriptor.path("null").getPath().get(0).getBytes(StandardCharsets.UTF_8)),
                CallOptions.timeout(FLIGHT_TIMEOUT, TimeUnit.SECONDS));
        if (actionResult.hasNext()) {
            Result result = actionResult.next();
            String result_body = new String(result.getBody(), StandardCharsets.UTF_8);
            return result_body.equals("yes");  /* Server returns "yes" if the results are ready, "no" otherwise */
        }
        return false;

    }


    /**
     * Initiates an action call to the connected server.
     * The action must be one of the servers possible actions seen in action list.
     * Responses from the server are printed in the console (TODO: change this if needed)
     * <p>
     * @param  actionType  a string representing the action the server will execute
     * @param  descriptor a string representing additional info needed by some actions, ignored if not needed
     */
    public void doAction(String actionType, String descriptor){
        Iterator<Result> actionResult = client.doAction(new Action(actionType,
                FlightDescriptor.path(descriptor).getPath().get(0).getBytes(StandardCharsets.UTF_8)));
        while (actionResult.hasNext()) {
            Result result = actionResult.next();
            System.out.println(new String(result.getBody(), StandardCharsets.UTF_8));
        }

    }
}
