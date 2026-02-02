package org.imsi.queryEREngine.imsi.er.Utilities;

import org.apache.arrow.flight.*;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ArrowFlightProducer implements FlightProducer {
    private final ArrowDataHandler dataSource;
    private final BufferAllocator allocator;
    private final ExecutorService executor;


    public ArrowFlightProducer(ArrowDataHandler dataSource, BufferAllocator allocator) {
        this.dataSource = dataSource;
        this.allocator = allocator;
        this.executor = Executors.newFixedThreadPool(10);
    }


    @Override
    public void getStream(CallContext callContext, Ticket ticket, ServerStreamListener listener) {

//        String type = extractFromTicket(ticket);
//        VectorSchemaRoot data;
//        if(type.equals("pair")){
//            data = dataSource.fetchPairs();
//        }
//        else if(type.equals("dict")){
//            data = dataSource.fetchDict();
//        }
//        else{
//            System.out.println("Unknown type passed in Flight server");
//            return;
//        }
//        System.out.println("extracted: " + type);
//        // Fetch data
//        try {
//            listener.start(data);
//            listener.putNext();
//            listener.completed();
//        } catch (Exception e) {
//            System.out.println("Error fetching data: " + e);
//        }

    }


    @Override
    public void listFlights(CallContext callContext, Criteria criteria, StreamListener<FlightInfo> listener) {

    }

    @Override
    public FlightInfo getFlightInfo(CallContext callContext, FlightDescriptor flightDescriptor) {
        return null;
    }

    @Override
    public Runnable acceptPut(CallContext context, FlightStream flightStream, StreamListener<PutResult> ackStream) {
        // Implement this method if you want to support clients uploading data
        // to your service
        return null;
    }

    @Override
    public void doAction(CallContext context, Action action, StreamListener<Result> listener) {
        // Implement this method if you want to support custom "actions"
    }

    @Override
    public void listActions(CallContext context, StreamListener<ActionType> listener) {
        // Implement this method to list the custom "actions" that this service supports
    }


    private String extractFromTicket(Ticket ticket) {
        String type = new String(ticket.getBytes());
        System.out.println("Type: " + type);
        return type;
    }


    private void printData(VectorSchemaRoot data) {
        // Print the schema
        Schema schema = data.getSchema();
        System.out.println("Schema: " + schema);

        // Print the field vectors and their data
        List<FieldVector> fieldVectors = data.getFieldVectors();
        System.err.println("SIZE: "+fieldVectors.size());
        for (FieldVector vector : fieldVectors) {
            System.out.println("Field: " + vector.getField().getName());
            System.out.println("Data: " + vector.getDataBuffer().toString());
        }
    }

    private void printFetchedData(VectorSchemaRoot data) {
        // Iterate over the rows or chunks of data and print the values
        int rowCount = data.getRowCount();
        List<FieldVector> fieldVectors = data.getFieldVectors();
        for (int row = 0; row < rowCount; row++) {
            System.out.println("Row: " + row);
            for (FieldVector vector : fieldVectors) {
                System.out.println("Field: " + vector.getField().getName());
                System.out.println("Value: " + vector.getObject(row));
            }
        }
    }
}
