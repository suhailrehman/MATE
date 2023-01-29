from collections import Counter, defaultdict
import pandas as pd
import pyarrow as pa
import numpy as np

from index_generation import XASH

from tqdm.auto import tqdm


def perform_ICS(query_df):
    all_cardinalities = query_df.apply(lambda x: x.nunique())
    return all_cardinalities.idxmin()


def generate_hash_superkey(row, hash_function=XASH, hash_size=256):
    superkey = 0
    value_list = []
    for colname, val in row.iteritems():
        value = str(val)
        superkey = superkey | hash_function(value, hash_size)
        value_list.append((colname, value))
    return superkey, value_list

def perform_initial_table_filtering(q_df, posting_list):
    init_column = perform_ICS(q_df)
    init_values = set(q_df[init_column].values.astype(str))

    initial_table_list = defaultdict(list)
    for value in tqdm(init_values, desc='Creating Initial Table List', leave=False):
        for tableid, rowid, colid, superkey in posting_list[value]:
            initial_table_list[tableid].append((value, rowid, colid, superkey))
            
    return sorted(initial_table_list.items(), key=lambda x: len(x[1]), reverse=True)


def perform_row_filtering(q_df, posting_list, table_pl, initial_table_list, superkeydf, init_col_label):
    ranked_table_list = []
    for tableid, table_pls in tqdm(initial_table_list, desc='Processing Inital Table List', leave=False):
        # TODO: Early stopping condition
        matching_rows = []
        for value, rowid, colid, superkey in tqdm(table_pls, desc="Processing Table PLs", leave=False):
            # Get list of unqiue superkeys for the value in the query table
            # TODO: Second early stopping condition based on num_rows checked
            init_row_match = superkeydf.loc[superkeydf[init_col_label] == value]
            for idx, sk_row in init_row_match.iterrows():
                if sk_row['superkey'] | superkey == superkey:
                     matching_rows.append((tableid, rowid, colid, value))
            
        ranked_table_list.append((tableid, matching_rows))
    
    return sorted(ranked_table_list, key=lambda x: len(x[1]), reverse=True)
    

    
def get_query_superkey(q_df, hash_function=XASH, hash_size=128):
    q_sk_df = q_df.copy()
    q_sk_df['superkey'] = q_sk_df.apply(lambda x: generate_hash_superkey(x, hash_function, hash_size)[0], axis=1)
    return q_sk_df


def process_query(q_df, posting_list, table_pl):
    superkeydf = get_query_superkey(q_df)
    init_table_list = perform_initial_table_filtering(q_df, posting_list)
    final_ranked_list = perform_row_filtering(q_df, posting_list, table_pl, init_table_list, superkeydf, perform_ICS(q_df))
    return [x[0] for x in final_ranked_list]

def process_gb_query(gb_query, posting_list, table_pl):
    q_df = pd.read_parquet(gb_query['dst_label'])[gb_query['args']['colset']]
    return process_query(q_df, posting_list, table_pl)

def process_pivot_query(pivot_query, posting_list, table_pl):
    pivot_dst = pd.read_parquet(pivot_query['dst_label'])
    id_var_name  = pivot_dst.index.name
    col_var_name = pivot_dst.columns.name

    query_colset = [id_var_name, col_var_name]

    q_df = pivot_dst.reset_index().melt(id_vars=id_var_name)[query_colset]
    return process_query(q_df, posting_list, table_pl)

def get_random_join_column_cardinality(df, colset=1):
    rand_left_col = set(np.random.choice([col for col in df.columns if '__LEFT' in col], colset))
    rand_right_col = set(np.random.choice([col for col in df.columns if '__RIGHT' in col], colset))
    return rand_left_col, rand_right_col

def construct_join_query_colsets(df, key_col, method=get_random_join_column_cardinality, colset=2):
    max_x_cols, max_y_cols = method(df, colset)
    return [set([key_col]).union(max_x_cols), set([key_col]).union(max_y_cols)]

def process_join_query(join_query, posting_list, table_pl):
    # Generate a new "unpivoted dataframe" to support existing query methods here
    key_col = join_query['args']['key_col']
    q_df = pd.read_parquet(join_query['dst_label'])
    colsets = construct_join_query_colsets(q_df, key_col)
    return [process_query(q_df[colset], posting_list, table_pl) for colset in colsets]
    




if __name__ == '__main__':
    
    import pickle

    server_serialized_file = "sketchset_256.ser"
    
    print('Loading Posting Lists')

    with open(server_serialized_file, 'rb') as fp:
        posting_list, table_pl = pickle.load(fp)
        
    query_output_dir='/tank/local/suhail/data/relic-datalake/gittables/outputs/100_queries/artifacts/'
    query_ops_file='/tank/local/suhail/data/relic-datalake/gittables/outputs/100_queries/operations.parquet'

    query_df = pd.read_parquet(query_ops_file)
    
    tqdm.pandas()
    
    #Groupby

    gb_queries = query_df.loc[(query_df.operation == 'groupby') & (query_df.colset_size == 2)]
    
    gb_results = gb_queries.copy()
    gb_results['results'] = gb_queries.progress_apply(lambda x: process_gb_query(x, posting_list, table_pl), axis=1)
    
    gb_results.to_parquet('gb_results_2col_MATE_256.parquet')
    
    
    #Pivots
    
    pivot_queries = query_df.loc[(query_df.operation == 'pivot')]
    
    pivot_results = pivot_queries.copy()
    pivot_results['results'] = pivot_queries.progress_apply(lambda x: process_pivot_query(x, posting_list, table_pl), axis=1)
    
    pivot_results.to_parquet('pivot_results_2col_MATE_256.parquet')
    
    
    # #Join Detection

    join_queries = query_df.loc[(query_df.operation == 'join')]
    
    join_results = join_queries.copy()
    join_results['results'] = join_queries.progress_apply(lambda x: process_join_query(x, posting_list, table_pl), axis=1)
    
    join_results.to_parquet('join_results_2col_MATE_256.parquet')

    
  
    