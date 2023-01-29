import argparse
import glob
from itertools import combinations
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple
#from attr import has
import pyarrow as pa
import pyarrow.flight
import pandas as pd
import time
from functools import reduce
from collections import defaultdict
import numpy as np
from sortedcontainers import SortedList

from tqdm.auto import tqdm
    
def get_max_join_column_cardinality(df, colset=1):
    all_cardinalities = df.apply(lambda x: x.nunique())
    return set(all_cardinalities.filter(regex='__LEFT').nlargest(colset).index), set(all_cardinalities.filter(regex='__RIGHT').nlargest(colset).index)

def get_random_join_column_cardinality(df, colset=1):
    rand_left_col = set(np.random.choice([col for col in df.columns if '__LEFT' in col], colset))
    rand_right_col = set(np.random.choice([col for col in df.columns if '__RIGHT' in col], colset))
    return rand_left_col, rand_right_col

def construct_join_query_colsets(df, key_col, method=get_random_join_column_cardinality, colset=2):
    max_x_cols, max_y_cols = method(df, colset)
    return [set([key_col]).union(max_x_cols), set([key_col]).union(max_y_cols)]


class FlightSketchClient:
    
    def __init__(self, location: str="grpc://0.0.0.0:33333", output_dir: str="./output/",
                 maxcolsetsize: int=2, max_chunksize: int=None, ranking_key='j_contain') -> None:
        self.location = location
        self.client = pa.flight.connect(location)
        self.max_colset_size = maxcolsetsize
        self.max_chunksize = max_chunksize
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.perf_records = []
        self.ranking_key = ranking_key


    def record_perf(self, op: str, label: str, start_time: float) -> None:
        self.perf_records.append(pd.Series({
                            'op': op,
                            'label': label,
                            'elapsed_time': time.perf_counter() - start_time
                            }).to_frame().T)
    
    def constuct_index_df(self, df: pd.DataFrame, df_label: str, index=False) -> None:
        """ 
        Constructs a sketch of the dataframe.
        """
               
        start_time = time.perf_counter()
        upload_table = pa.table(df)
        upload_descriptor = pa.flight.FlightDescriptor.for_path(df_label.encode('utf-8'))
        batches = upload_table.to_batches(max_chunksize=self.max_chunksize)
        writer, _ = self.client.do_put(upload_descriptor, batches[0].schema)
        with writer:
            for batch in batches:
                writer.write_batch(batch)
        self.record_perf('MATE_index', df_label, start_time)

    def process_results(self, results) -> List[str]:
        #print(results)
        if not results:
            return []
        return [x.body.to_pybytes().decode('utf-8') for x in results][0].split('\n')
    
    
    def query_single_col(self, mh_label, hash_list) -> List[str]:
        # Upload Query Table First
        start_time = time.perf_counter()
        upload_table = pa.table({mh_label: hash_list})
        query_descriptor = pa.flight.FlightDescriptor.for_path(mh_label.encode('utf-8'))
        batches = upload_table.to_batches(max_chunksize=self.max_chunksize)
        writer, _ = self.client.do_put(query_descriptor, batches[0].schema)
        with writer:
            for batch in batches:
                writer.write_batch(batch)
        self.record_perf('query_hash_upload', mh_label, start_time)  
        
        # Query sketch for the uploaded table
        start_time = time.perf_counter()
        results = self.process_results(self.client.do_action(pa.flight.Action("QUERY", mh_label.encode('utf-8'))))
        self.record_perf('query_time', mh_label, start_time)
        return results
    
    def dictify_results(self, results: Dict[str, SortedList]) -> str:
        return_dict = {}
                
        for colset, sortedlist in results.items():
            return_dict[colset] = {}
            for key, items in sortedlist.items():
                return_dict[colset][key] = []
                for item in items:
                    return_dict[colset][key].append(item)
        
        return return_dict

    def query_sketch(self, df, df_label, colset: set, rank=True) -> Dict[str, SortedList]:
        """
        Queries the server index for the given column set.
        If the column set > max_colset_size, then it is split into multiple column sets and queried independently.
        Results are then merged (via set intersection) to get the final result.
        """
        print(f'Querying {df_label}, {colset}')
        start_time = time.perf_counter()
        results = {}
        query_dict = {}
        if len(colset) > self.max_colset_size:
            print(colset)
            #colsets = list(colset)
            for single_colset in combinations(colset,self.max_colset_size):
                mh_label, hash_list = construct_sketch_colset(df, "##QUERY##"+df_label, single_colset)
                query_dict[mh_label] = hash_list
                
        else:    
            mh_label, hash_list = construct_sketch_colset(df, "##QUERY##"+df_label, colset)
            query_dict[mh_label] = hash_list
            
        self.record_perf('mmh3_compute_query', df_label, start_time)

        
        for mh_label, hash_list in query_dict.items():
            results[mh_label] = self.initialize_ranked_list()
            q_result = set(self.query_single_col(mh_label, hash_list))
            for result in q_result:
                if result and result.strip():
                    try:
                        #print(result, mh_label)
                        jc_result = self.compute_jc(result, mh_label)
                        #print(result, mh_label, jc_result)
                        results[mh_label].add(jc_result)
                    except pa.lib.ArrowKeyError as e:
                        print('KeyError:', result)
                        continue
                    
        #print('Results:', results)
        # if results:               
        #     if len(results) > 1 and self.max_colset_size == 1:
        #         return self.merge_results(results)
        #     else:
        #         return list(results[0])
        # else:
        #     return []
        
        return results
    
    def index_all(self) -> List[str]:
        start_time = time.perf_counter()
        results = self.process_results(self.client.do_action(pa.flight.Action("INDEXALL", 'INDEXALL'.encode('utf-8'))))
        self.record_perf('indexing_time', 'all', time.perf_counter())
        return results

    def process_filelist(self, file_list: List[str], index=False, index_all=True) -> None:
        for file in tqdm(file_list, desc='Processing Files in FileList', leave=False):
            table_file = os.path.basename(file)
            df = None
            start_time = time.perf_counter()
            if file.endswith('.csv'):
                df = pd.read_csv(file, index_col=0) 
            elif file.endswith('.parquet'):
                df = pd.read_parquet(file)
            
            if df is not None:
                self.record_perf('read',table_file, start_time)
                start_time = time.perf_counter()
                self.constuct_index_df(df, table_file, index=index)
                self.record_perf('sketch_finish',table_file, start_time)

    def process_directory(self, in_dir: str, index=False, index_all=True) -> None:
        file_list = []
        for ext in ('*.csv', '*.parquet'):
            file_list.extend(glob.glob(os.path.join(in_dir, ext)))

        self.process_filelist(file_list, index=index, index_all=True)

    """
    def query_index_pivot_old(self, q_df: pd.DataFrame, q_df_name):
        # Generate a new "unpivoted dataframe" to support existing query methods here
        index_col_name = q_df.index.name
        temp_df = q_df.melt(ignore_index=False).reset_index()
        column_col_name = temp_df.columns[1]
        value_col_name = 'value'

        index_results = set(self.query_sketch(temp_df, q_df_name, [index_col_name]))
        column_results = set(self.query_sketch(temp_df, q_df_name, [column_col_name]))
        value_results = set(self.query_sketch(temp_df, q_df_name, [value_col_name]))
        
        return list(index_results.union(column_results).union(value_results))
    """
    
    def query_index_pivot(self, pivot_dst: pd.DataFrame, q_df_name: str):
        id_var_name  = pivot_dst.index.name
        col_var_name = pivot_dst.columns.name

        query_colset = [id_var_name, col_var_name]

        q_df = pivot_dst.reset_index().melt(id_vars=id_var_name)
        return {str(query_colset): self.query_sketch(q_df, q_df_name, query_colset)}
    
    def query_join(self, q_df: pd.DataFrame, q_df_name, key_col):
        # Generate a new "unpivoted dataframe" to support existing query methods here
        colsets = construct_join_query_colsets(q_df, key_col)
        results = {}
        for colset in colsets:
            results[str(colset)] = self.query_sketch(q_df, q_df_name, colset)
        
        return results

    def query_directory(self, query_df: pd.DataFrame, result_file='result.parquet'):
        query_results = []
        os.makedirs(f"{self.output_dir}/ranking_results/", exist_ok=True)
        for _, row in query_df.iterrows():
            try:
                dst_file = row['dst_label']
                table_file = os.path.basename(dst_file)
                dst_df = None
                start_time = time.perf_counter()
                if dst_file.endswith('.csv'):
                    dst_df = pd.read_csv(dst_file, index_col=0) 
                elif dst_file.endswith('.parquet'):
                    dst_df = pd.read_parquet(dst_file)
                
                if dst_df.empty:
                    continue
            
                op_type = row['operation']
                if op_type == 'groupby':
                    colset = row['args']['colset']
                    # Query the index
                    results = {str(colset) : self.query_sketch(dst_df, table_file, colset)}
                elif op_type == 'join':
                    # colset = row['args']['key_col']
                    # Query the index
                    # results = self.query_sketch(dst_df, table_file, [colset])
                    results = self.query_join(dst_df, table_file, row['args']['key_col'])
                elif op_type == 'pivot':
                    results = self.query_index_pivot(dst_df, table_file)
                
                # Write the results to file
                ranking_result_file = f"{self.output_dir}/ranking_results/{_}.pkl"
                with open(ranking_result_file, 'wb') as f:
                    pickle.dump(self.dictify_results(results), f)
                query_results.append(ranking_result_file)
            except FileNotFoundError as e:
                print(f"Not Found: {row['dst_label']}")
                continue
            except ValueError as e:
                print(f"ValueError: {row['dst_label']}")
                print(e)
                raise e

        result_df = query_df.copy()
        result_df['results'] = pd.Series(query_results)
        result_df.to_parquet(f"{self.output_dir}/{result_file}")
        print('Results written to:', f"{self.output_dir}/{result_file}")

        return result_df


    def change_threshold(self, threshold: float =0.5) -> None:
        return self.process_results(self.client.do_action(pa.flight.Action("THRESHOLD", str(threshold).encode('utf-8'))))

    def write_perf_file(self, filename='performance.csv') -> None:
        if self.perf_records:
            perf_filename= f"{self.output_dir}/{filename}"
            pd.concat(self.perf_records, ignore_index=True).to_csv(perf_filename, mode='a', header= not os.path.exists(perf_filename))     
    
    def compute_jc(self, key: str, q_key: str):
        start_time = time.perf_counter()
        results = list(self.client.do_action(pa.flight.Action("JC_MINHASH", f"{key};;;;{q_key}".encode('utf-8'))))
        res_string = results[0].body.to_pybytes().decode('utf-8')
        res_dict = json.loads(res_string)
        res_dict.update({'key': key, 'q_key': q_key})
        self.record_perf('compute_jc',f'{key};;;;{q_key}', start_time)
        return res_dict

    def serialize_sketches(self, filename='sketchset.ser') -> None:
        start_time = time.perf_counter()
        self.client.do_action(pa.flight.Action("SERIALIZE", filename.encode('utf-8')))
        self.record_perf('serialize_time', 'all', start_time)

    def load_sketches(self, filename='sketchset.ser') -> None:
        start_time = time.perf_counter()
        self.client.do_action(pa.flight.Action("LOAD", filename.encode('utf-8')))
        self.record_perf('load_time', 'all', start_time)
        
    def initialize_ranked_list(self) -> SortedList:
        return SortedList(key=lambda x: -x[self.ranking_key])
        
    def merge_results(self, results: List[set], query_key=None) -> List:
        merged_dicts = [defaultdict(set)] * len(results)
            
        for ix, resultset in enumerate(results):
            for res in resultset:
                try:
                    df_label = res.split('//')[0]
                    df_columns = res.split('//')[1].split(';;')
                    merged_dicts[ix][df_label].update(df_columns)
                except:
                    continue
                
        common_df_labels = set(merged_dicts[0].keys())
        
        for ix, dflabeldict in enumerate(merged_dicts[1:]):
            common_df_labels.intersection(set(merged_dicts[ix].keys()))
        
        merged_results = self.initialize_ranked_list()
        for df_label in common_df_labels:
            colset = set()
            for ix in range(len(merged_dicts)):
                colset = colset.union(merged_dicts[ix][df_label])
            #print(colset)
            for cset in colset:
                merged_results.add(self.compute_jc(df_label+"//"+cset, query_key))       
        return merged_results
        

def load(sk_client: FlightSketchClient, filename: str) -> None:
    #result_dir = sys.argv[3]
    #sk_client.output_dir = result_dir
    sk_client.load_sketches(filename=filename)
    sk_client.index_all()
    sk_client.write_perf_file()

def sketch_and_serialize(sk_client: FlightSketchClient, file_path: str, index=False) -> None:
    #file_path = sys.argv[2]
    #result_dir = sys.argv[3]
    #sk_client.output_dir = result_dir
    #Check if path is file or directory
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fp:
            file_list = [eval(x) for x in fp.readlines()]
            sk_client.process_filelist(file_list, index=index)
    elif os.path.isdir(file_path):
        sk_client.process_directory(file_path, index=index)
    
    sk_client.write_perf_file()
    #sk_client.index_all()
    #sk_client.write_perf_file()
    sk_client.serialize_sketches()
    #sk_client.write_perf_file()

def query(sk_client: FlightSketchClient, query_file: str) -> None:
    # Load query dataframe
    #query_dir = sys.argv[2]
    #result_dir = sys.argv[4]
    query_df = pd.read_parquet(query_file)

    result_df = sk_client.query_directory(query_df)
    sk_client.write_perf_file()
    

    print(result_df)

def threshold(sk_client: FlightSketchClient, new_threshold: float) -> None:
    # Load query dataframe
    results = sk_client.change_threshold(new_threshold)
    print(f"Threshold Change Result: {results}")


def parse_args():
    parser = argparse.ArgumentParser(description='DataSketch Client for Apache Arrow Server')
    parser.add_argument('--server', type=str, default='grpc://0.0.0.0:33333', help='Server protocol://address:port')
    parser.add_argument('--mode', type=str, default='sketch', help='Mode of operation (sketch | load | query | threshold)')
    parser.add_argument('--input', type=str, help='File to load (Parquet File List / Directory or Serialized Sketch File)')
    parser.add_argument('--query_file', type=str, help='Query file')
    parser.add_argument('--result_dir', type=str, help='Result Directory for Performance and Accuracy Files')
    parser.add_argument('--threshold', type=float, help=' Set new Threshold (Triggers Reindex)')
    parser.add_argument('--maxcolset', type=int, default=2, help='Max Column Set Size')
    parser.add_argument('--max_chunksize', type=int, default=None, help='Max Chunk Size')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    sk_client = FlightSketchClient(args.server, args.result_dir, args.maxcolset, args.max_chunksize)
    if args.mode == 'sketch':
        sketch_and_serialize(sk_client, args.input)
    elif args.mode == 'load':
        load(sk_client, args.input)
    elif args.mode == 'query':
        query(sk_client, args.query_file)
    elif args.mode == 'threshold':
        threshold(sk_client, args.threshold)
    else:
        print("Invalid command")
        exit(1)

