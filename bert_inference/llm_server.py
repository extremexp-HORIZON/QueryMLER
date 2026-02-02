#from concurrent.futures.process import _threads_wakeups
import pyarrow.flight as flight
from time import sleep
from bert_infer import inference

class TinyServer(flight.FlightServerBase):

    def __init__(self, 
                 host = '0.0.0.0', 
                 port = 5678):
        self.tables = {}
        self.ready = False
        self.location = flight                  \
                        .Location               \
                        .for_grpc_tcp(host, port)
        super().__init__(self.location)

    def do_put(self, context, descriptor, reader, 
               writer):
        table_name = descriptor.path[0].decode('utf-8')
        #print(table_name) 
        self.tables[table_name] = reader.read_all()
        #print(self.tables[table_name])


    def do_get(self, context, ticket):
        table_name = ticket.ticket.decode('utf-8')
        table = self.tables[table_name]
        print(table)
        return flight.RecordBatchStream(table)
  
    def flight_info(self, descriptor):
        table_name = descriptor.command
        table = self.tables[table_name]

        ticket = flight.Ticket(table_name)
        location = self.location.uri.decode('utf-8')
        endpoint = flight.FlightEndpoint(ticket,
                                         [location])
        
        return flight.FlightInfo(table.schema, 
                                 descriptor, 
                                 [endpoint], 
                                 table.num_rows,
                                 table.nbytes)
    
    def get_flight_info(self, context, descriptor):
        table_name = descriptor.command
        return self.flight_info(descriptor)        
        
    def list_flights(self, context, criteria):
        for table_name in self.tables.keys():
            descriptor = flight                  \
                         .FlightDescriptor       \
                         .for_command(table_name)
            yield self.flight_info(descriptor)

    def do_action(self, context, action):
        # Add flight.Result instances you want to return in this list
        results = []
        if action.type == 'drop_table':
            table_name = action.body.to_pybytes()
            del self.tables[table_name]
        elif action.type == 'is_ready':
            results = self.is_ready()
        elif action.type == 'bert_inference':
            # Get candidates file path from action body
            candidates_file = action.body.to_pybytes().decode('utf-8')
            if candidates_file == 'null' or not candidates_file:
                candidates_file = '/data/candidates.csv'
            results = self.bert_inference(candidates_file)
            self.ready = True
        elif action.type == 'shutdown':
            self.shutdown()
        else:
            raise KeyError('Unknown action {!r}'.
                           format(action.type))
        
        return results

    def list_actions(self, context):
        return [('drop_table', 'Drop table'),
                ('is_ready', 'Responds yes if results are ready, no otherwise'),
                ('shutdown', 'Shut down server'),]

    def is_ready(self):
        results = []
        if(self.ready):
            result = flight.Result("yes".encode('utf-8'))
            results.append(result)
        else:
            result = flight.Result("no".encode('utf-8'))
            results.append(result)
        return results
    
    def bert_inference(self, candidates_file='/data/candidates.csv'):
        results = []
       # pairs_arrow = self.tables['pairs']
        dict_arrow = self.tables['dict']
        eqbi_arrow = self.tables['eqbi']
        print(self.tables['eqbi'])
        print(f"Running inference with candidates file: {candidates_file}")
        output = inference(dict_arrow, eqbi_arrow, candidates_file)
        self.tables['results'] = output
        results.append(flight.Result("success".encode('utf-8')))
        return results
        

if __name__=='__main__':
    server = TinyServer(port=5678)
    server.serve()