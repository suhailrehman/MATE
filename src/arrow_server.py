from collections import Counter, defaultdict
import json
import pathlib
import pickle
import math

import pyarrow as pa
import pyarrow.flight
from traitlets import default
import numpy as np

# from datasketch import MinHashLSHEnsemble, MinHash, LeanMinHash
from index_generation import XASH



class FlightServer(pa.flight.FlightServerBase):

    def __init__(self, location="grpc://0.0.0.0:33333",
                 repo=pathlib.Path("./datasets"), default_threshold=0.5,
                 num_perm=256,
                 **kwargs):
        super(FlightServer, self).__init__(location, **kwargs)
        self._location = location
        self._repo = repo
        # self.hashes = {}
        # self.query_hashes = {} # Dict of hashes specifically to query (don't index)
        # self.threshold=default_threshold
        # self.num_perm = num_perm
        # self.lshe_index = MinHashLSHEnsemble(threshold=default_threshold, num_perm=num_perm)
        self.posting_list = defaultdict(list)
        self.table_pl = {}
        
        print(f'MATE server running in {location}')
        
        
    def _make_flight_info(self, dataset):
        dataset_path = self._repo / dataset
        schema = pa.parquet.read_schema(dataset_path)
        metadata = pa.parquet.read_metadata(dataset_path)
        descriptor = pa.flight.FlightDescriptor.for_path(
            dataset.encode('utf-8')
        )
        endpoints = [pa.flight.FlightEndpoint(dataset, [self._location])]
        return pa.flight.FlightInfo(schema,
                                        descriptor,
                                        endpoints,
                                        metadata.num_rows,
                                        metadata.serialized_size)

    def list_flights(self, context, criteria):
        for dataset in self._repo.iterdir():
            yield self._make_flight_info(dataset.name)

    def get_flight_info(self, context, descriptor):
        return self._make_flight_info(descriptor.path[0].decode('utf-8'))

    def generate_hash_superkey(self, row, hash_function=XASH, hash_size=256):
        superkey = 0
        value_list = []
        for colname, val in row.iteritems():
            value = str(val)
            superkey = superkey | hash_function(value, hash_size)
            value_list.append((colname, value))
        return superkey, value_list

    def index_dataframe(self, df):
        for idx, row in df.iterrows():
            superkey, value_list = self.generate_hash_superkey(row)
            yield idx, superkey, value_list
            
    def update_postinglist(self, posting_list):
        for key, value in posting_list.items():
            self.posting_list[key].extend(value)
            
            
    def generate_posting_list(self, df, label='label'):
        posting_list = defaultdict(list)
        for idx, superkey, value_list in self.index_dataframe(df):
            for colname, value in value_list:
                posting_list[value].append((label, idx, colname, superkey))
        self.update_postinglist(posting_list)
        return posting_list

    def do_put(self, context, descriptor, reader, writer):
        # Get an entire table and generate XHASH 
        df_label = descriptor.path[0].decode('utf-8')
        # Read the uploaded data and write to Parquet incrementally
        # print('Server: PUT:', df_label)
        
        for chunk in reader:
            current_df = pa.Table.from_batches([chunk.data]).to_pandas()
            posting_list = self.generate_posting_list(current_df, label=df_label)
            self.table_pl[df_label]  =  posting_list
            self.update_postinglist(posting_list)


    def do_get(self, context, ticket):
        dataset = ticket.ticket.decode('utf-8')
        # TODO: Stream back minhashes to client after splitting dataset 
        # key and find all matching hashes
        
        # Stream data from a file
        dataset_path = self._repo / dataset
        reader = pa.parquet.ParquetFile(dataset_path)
        return pa.flight.GeneratorStream(
            reader.schema_arrow, reader.iter_batches())

    def minhash_jc(self, key, q_key):
        """ Jaccard Similarity and Containment between two MinHash objects in Hashes Dict """
        k_mh, k_len = self.hashes[key]
        q_mh, q_len = self.query_hashes[q_key]
        
        j_sim = k_mh.jaccard(q_mh)
        j_contain = (k_len/q_len + 1) * j_sim / (1 + j_sim)
        
        return j_sim, j_contain

    def list_actions(self, context):
        return [
            ("INDEX", "Index a specific dataset"),
            ("INDEXALL", "Index all current non-query hashes"),
            ("THRESHOLD", "Change LSH Index Threshold (triggers reindex for LSH-E)"),
            ("QUERY", "Query the index for a specific key (must already be uploaded via PUT)"),
            ("SERIALIZE", "Pickle the current Server Object to file (Hashes and Indexes)"),
            ("LOAD", "Unpickle the current Server Object from file (Hashes and Indexes)")   
        ]
        
    def index_all(self) -> None:
        del(self.lshe_index)
        self.lshe_index = MinHashLSHEnsemble(threshold=self.threshold, num_perm=self.num_perm) 
        self.lshe_index.index([(key, valueset, size)
                                    for key, (valueset, size) in self.hashes.items()])

    def do_action(self, context, action):
        result = []
        #print(f'Action to be Performed: {action.type} : {action.body.to_pybytes().decode("utf8")}')
        if action.type == "INDEX":
            result.append(f'Cannot index individual items in LSH-Ensemble')
        elif action.type == "INDEXALL":
            self.index_all()
            result.append(f'Indexed {len(self.hashes)} items at threshold {self.threshold}')
        elif action.type == "THRESHOLD":
            self.threshold = float(action.body.to_pybytes().decode('utf-8'))
            self.index_all()
            result.append(f'Indexed {len(self.hashes)} items at threshold {self.threshold}')
        elif action.type == "QUERY":
            query_key = action.body.to_pybytes().decode('utf-8')
            print(f'Query Key: {query_key}')
            query_mh, query_len = self.query_hashes[query_key]
            query_results = list(self.lshe_index.query(query_mh, query_len))
            #print(f'# Query Results: {len(query_results)}')
            result = ['\n'.join(query_results)]
        elif action.type == "SERIALIZE":
            to_path = action.body.to_pybytes().decode('utf-8')
            with open(to_path, 'wb') as fp:
                pickle.dump((self.posting_list, self.table_pl), fp)
            result.append(f'Written out minhashes to {to_path}')
        elif action.type == "LOAD":
            from_path = action.body.to_pybytes().decode('utf-8')
            with open(from_path, 'rb') as fp:
                self.posting_list, self.table_pl = pickle.load(fp)
            result.append(f'Loaded minhashes from {from_path}')
        elif action.type == "JC_MINHASH":
            key, q_key = action.body.to_pybytes().decode('utf-8').split(";;;;")
            print(f'JC_MINHASH: {key}, {q_key}')
            j_sim, j_contain = self.minhash_jc(key, q_key)
            print(j_sim, j_contain)
            result.append(json.dumps({"j_sim": j_sim, "j_contain": j_contain}))
        else:
            raise NotImplementedError
        return [x.encode('utf8') for x in result]


if __name__ == '__main__':
    server = FlightServer()
    server._repo.mkdir(exist_ok=True)
    server.serve()