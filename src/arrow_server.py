from collections import defaultdict
import json
import pathlib
import pickle

import pyarrow as pa
import pyarrow.flight
from traitlets import default

from datasketch import MinHashLSHEnsemble, MinHash, LeanMinHash

class FlightServer(pa.flight.FlightServerBase):

    def __init__(self, location="grpc://0.0.0.0:33333",
                 repo=pathlib.Path("./datasets"), default_threshold=0.5,
                 num_perm=256,
                 **kwargs):
        super(FlightServer, self).__init__(location, **kwargs)
        self._location = location
        self._repo = repo
        self.hashes = {}
        self.query_hashes = {} # Dict of hashes specifically to query (don't index)
        self.threshold=default_threshold
        self.num_perm = num_perm
        self.lshe_index = MinHashLSHEnsemble(threshold=default_threshold, num_perm=num_perm)
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

    def hash_row_vals(self, hash_function: Any, row: Any, hash_size: int) -> None:
        """Calculates Hash value for row.

        Parameters
        ----------
        hash_function : Any
            Hash function to use for hash calculation.

        row : Any
            Input row.

        hash_size : int
            Number of bits.

        Returns
        -------
        int
            Hash value for row.
        """
        hresult = 0
        for q in row.columns:
            d, hvalue = hash_function(row[q], hash_size)
            hresult = hresult | hvalue
        return hresult

    def do_put(self, context, descriptor, reader, writer):
        # dataset = descriptor.path[0].decode('utf-8')
        # # Read the uploaded data and write to Parquet incrementally
        # unique_value_dict = defaultdict(set)
        # #print('Server: PUT:', dataset)
        
        # for chunk in reader:
        #     current_df = pa.Table.from_batches([chunk.data]).to_pandas()
        #     for column in current_df.columns:
        #         for value in current_df[column].values:
        #             unique_value_dict[column].add(value.encode('utf-8'))
                    
        # for key, valueset in unique_value_dict.items():
        #     mh_obj = MinHash(num_perm=self.num_perm)
        #     mh_obj.update_batch(list(valueset))
        #     if '##QUERY##' in dataset:
        #         self.query_hashes[key] = (LeanMinHash(mh_obj), len(valueset))
        #     else:
        #         #print('STORING: ', key)
        #         self.hashes[key] = (LeanMinHash(mh_obj), len(valueset))

        # Get an entire table and generate XHASH 
        dataset = descriptor.path[0].decode('utf-8')
        # Read the uploaded data and write to Parquet incrementally
        unique_value_dict = defaultdict(set)
        #print('Server: PUT:', dataset)
        
        for chunk in reader:
            current_df = pa.Table.from_batches([chunk.data]).to_pandas()
            
                    
        for key, valueset in unique_value_dict.items():
            mh_obj = MinHash(num_perm=self.num_perm)
            mh_obj.update_batch(list(valueset))
            if '##QUERY##' in dataset:
                self.query_hashes[key] = (LeanMinHash(mh_obj), len(valueset))
            else:
                #print('STORING: ', key)
                self.hashes[key] = (LeanMinHash(mh_obj), len(valueset))




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
                pickle.dump(self.hashes, fp)
            result.append(f'Written out minhashes to {to_path}')
        elif action.type == "LOAD":
            from_path = action.body.to_pybytes().decode('utf-8')
            with open(from_path, 'rb') as fp:
                self.hashes = pickle.load(fp)
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