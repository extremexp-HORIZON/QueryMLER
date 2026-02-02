import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification
from torch import nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pyarrow
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import string
import re
from collections import Counter
import time
import os

EPOCHS = 5
MODEL_NAME = "./my_distilbert_model"
TOKENIZER_NAME = "./my_distilbert_tokenizer"
BATCH_SIZE = 32

# The two arguements are in arrow format
# candidates_file: optional path where to save the candidates CSV (default: /data/candidates.csv)
def inference(dict_arrow, eqbi_arrow, candidates_file="/data/candidates.csv"):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     # Initiate the dataframe that we will input in the model
#     test_df = pd.DataFrame(columns=['id1','id2'])

#     # Convert into pandas 
#  #   pairs_df = pairs_arrow.to_pandas()
    print(eqbi_arrow)
    dict_df = dict_arrow.to_pandas()
    print(dict_df)
    eqbi_df = eqbi_arrow.to_pandas()
    print(eqbi_df)
    #print('Amount of pairs: ' + str(pairs_df.shape[0]))
    print('Amount of unique entries: ' + str(dict_df.shape[0]))

    # Create lists to hold row, column, and value data
    rows = []
    columns = []
    values = []
    max_col = 0

    eqbi_df['values'] = eqbi_df['values'].apply(lambda x: [int(val) for val in x])

    # Iterate over the DataFrame rows to populate row, column, and value lists
    for index, row in eqbi_df.iterrows():
        key = row['key']
        key_index = index
        for value in row['values']:
            rows.append(key_index)
            columns.append(value)
            values.append(1)  # Assuming you want a value of 1 for each entry
            max_col = max(max_col, value + 1)

    # Convert lists to tensors
    rows_t = torch.tensor(rows)
    columns_t = torch.tensor(columns)

    # Create the PyTorch sparse tensor using torch.sparse_coo_tensor
    indices_t = torch.stack((rows_t, columns_t))
    values_t = torch.tensor(values, dtype=torch.float)  # Assuming values are float

    # Get the number of rows for the sparse tensor size
    num_rows = len(eqbi_df)
    size = (num_rows, max_col)

    # Create the sparse tensor
    X_gpu = torch.sparse_coo_tensor(indices_t, values_t, size=size)
    X_gpu = X_gpu.coalesce()
    print(X_gpu)
    X_gpu = X_gpu.to(device)

    # block-purging
    block_purging_start = time.time()

    indices = X_gpu._indices()
    values = X_gpu._values()
    size = X_gpu.size()

    rows_to_keep = indices[0] > 25
    new_indices = indices[:, rows_to_keep]
    new_values = values[rows_to_keep]

    X_gpu = torch.sparse_coo_tensor(new_indices, new_values, size=size)

    block_purging_end= time.time()
    print(f"Block Purging finished in: {block_purging_end - block_purging_start} seconds")

    print(f"Current GPU memory allocated 4: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    del indices, values, rows_to_keep, new_indices, new_values

    block_filtering_start= time.time()
    # sum along columns
    sum_columns = torch.sparse.sum(X_gpu, dim=0)

    dense_sum_columns = sum_columns.to_dense()
    rounded_result = torch.round(dense_sum_columns * 0.5)
    X_gpu = X_gpu.coalesce()

    def cumulative_count_vector(input_vector):
        unique_numbers, counts = torch.unique(input_vector, return_counts=True)
        cumulative_dict = {num: 0 for num in unique_numbers.tolist()}

        def get_cumulative_count(num):
            nonlocal cumulative_dict
            result = cumulative_dict[num]
            cumulative_dict[num] += 1
            return result
        
        result_vector = torch.tensor(list(map(get_cumulative_count, input_vector.tolist())))
        return result_vector

    print(f"Current GPU memory allocated 5: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    X_gpu = X_gpu.coalesce()
    colindxs = X_gpu.indices()[1]
    cumsums = cumulative_count_vector(colindxs)
    cumsums = cumsums.to(device)
    values = X_gpu.values()
    values += cumsums
    X_gpu._values().set_(values)
    X_gpu = X_gpu.coalesce()

    X_gpu = X_gpu.coalesce()

    indices = X_gpu.indices()
    values = X_gpu.values()
    thresholds = rounded_result[indices[1]]
    result_values = torch.where(values > thresholds, torch.tensor(1, device=device), torch.tensor(0, device=device))

    step_function_result = torch.sparse_coo_tensor(indices, result_values, size=X_gpu.size())

    def remove_explicit_zeros(input_sparse_tensor):
        indices = input_sparse_tensor._indices()
        values = input_sparse_tensor._values()
        non_zero_mask = values != 0
        non_zero_indices = indices[:, non_zero_mask]
        non_zero_values = values[non_zero_mask]
        non_zero_result = torch.sparse_coo_tensor(non_zero_indices, non_zero_values, size=input_sparse_tensor.size())
        return non_zero_result

    step_function_result = remove_explicit_zeros(step_function_result)
    step_function_result = step_function_result.coalesce()

    block_filtering_end= time.time()
    print(f"Block Filtering finished in: {block_filtering_end - block_filtering_start} seconds")

    del rounded_result, colindxs, cumsums, values, indices, thresholds, result_values, X_gpu
    print(f"Current GPU memory allocated 6: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    edge_pruning_start= time.time()
    X_transposed = step_function_result.t().to(dtype=torch.float16)
    X_transposed = X_transposed.coalesce()
    batch_indices = None
    batch_values = None
    k_threshold = len(step_function_result.values())

    sum_columns = torch.sparse.sum(step_function_result, dim=0)
    dense_sum_columns = sum_columns.to_dense().float()

    #batch_size = 16384 * 4 / torch.mean(dense_sum_columns, dim=0).item()
    batch_size = 2000
    batch_size = round(batch_size)
    print("Batch size is: ", batch_size)

    topk_values = torch.tensor([], dtype=torch.float16, device=device)
    topk_indices = torch.tensor([], dtype=torch.float16, device=device)
    del step_function_result
    batch_counter = 0

    for i in range(0, X_transposed.size()[0], batch_size):
        X_transposed = X_transposed.coalesce()
        batch_counter += 1
        
        row_indices = X_transposed.indices()[1]
        column_indices = X_transposed.indices()[0]
        mask = torch.logical_and(i <= column_indices, column_indices < i + batch_size)
        selected_columns = column_indices[mask]
        selected_rows = row_indices[mask]

        values_batch = X_transposed.values()[mask]
        selected_indices = torch.stack((selected_rows, selected_columns))
        if batch_counter <= 2:
            print(f"Current GPU memory allocated 7: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        batch_step_function_result = torch.sparse_coo_tensor(
            indices=selected_indices,
            values=values_batch,
            size=(X_transposed.size()[1], X_transposed.size()[0]),
            device=X_transposed.device,
            dtype=torch.float16
        )
        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        X_transposed = X_transposed.to(dtype=torch.float32)
        batch_step_function_result = batch_step_function_result.to(dtype=torch.float32)
        batch_result = torch.sparse.mm(X_transposed, batch_step_function_result).to(dtype=torch.int8)

        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8.1: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    #  batch_result = batch_result.coalesce()
        
        batch_values_c = batch_result._values().clone()
        batch_row_indices = batch_result._indices()[0]
        batch_column_indices = batch_result._indices()[1]
            
        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8.2: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        del batch_result
        batch_row_indices = batch_row_indices.to(dtype=torch.int32)
        batch_column_indices = batch_column_indices.to(dtype=torch.int32)
        # batch_row_indices = batch_result._indices()[0].to(dtype=torch.int32)
        # batch_column_indices = batch_result._indices()[1].to(dtype=torch.int32)
        
        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8.22: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        # remove the entities of this batch from X_transposed
        X_row_indices = X_transposed.indices()[0]
        X_column_indices = X_transposed.indices()[1]
        X_mask = (X_row_indices >= i + batch_size)
        remaining_rows = X_row_indices[X_mask]
        remaining_columns = X_column_indices[X_mask]
        remaining_values = X_transposed.values()[X_mask]
        remaining_indices = torch.stack((remaining_rows, remaining_columns))

        X_transposed = torch.sparse_coo_tensor(
            indices=remaining_indices,
            values=remaining_values,
            size=X_transposed.size(),
            device=X_transposed.device
        )

        del X_mask, X_row_indices, X_column_indices, remaining_rows, remaining_columns, remaining_values, remaining_indices

        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8.3: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        # from the resulting matrix keep only relevant values (batch)
    #  batch_result._indices()[:, :] = batch_result._indices().to(dtype=torch.int32)

        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8.4: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        batch_mask = torch.logical_and(torch.logical_and(i <= batch_column_indices, batch_column_indices < i + batch_size), batch_row_indices > batch_column_indices)
        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8.5: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        batch_selected_columns = batch_column_indices[batch_mask]
        
        del batch_column_indices
        if batch_counter <= 2:
            print(f"Current GPU memory allocated 8.6: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        batch_selected_rows = batch_row_indices[batch_mask].to(dtype=torch.int32)
        del batch_row_indices
        
        batch_values_batch = batch_values_c[batch_mask].to(dtype=torch.int8)
        batch_selected_indices = torch.stack((batch_selected_rows, batch_selected_columns)).to(dtype=torch.long)
        
        del batch_mask, batch_selected_rows, batch_selected_columns
        # batch_result._indices()[:, :] = batch_selected_indices
        # batch_result._values() = batch_values_batch

        common_blocks = batch_values_batch
        log_term_i = torch.log10(X_transposed.size(1) / dense_sum_columns[batch_selected_indices[0]])
        log_term_j = torch.log10(X_transposed.size(1) / dense_sum_columns[batch_selected_indices[1]])

        modified_values = common_blocks * log_term_i * log_term_j

        del common_blocks, log_term_i, log_term_j
        if batch_counter <= 2:
            print(f"Current GPU memory allocated 9: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        # print(len(modified_values))
        # print(len(batch_selected_indices[0]))
        if len(modified_values) > k_threshold:
            topk_batch_values, topk_batch_flat_indices = modified_values.topk(k=k_threshold)
            if batch_indices is None:
                batch_indices = batch_selected_indices[:, topk_batch_flat_indices]
            else:
                batch_indices = torch.cat((batch_indices, batch_selected_indices[:, topk_batch_flat_indices]), dim=1)        
            if batch_values is None:
                batch_values = topk_batch_values
            else:
                batch_values = torch.cat((batch_values, topk_batch_values), dim=0)
            batch_values, topk_indices = batch_values.topk(k=k_threshold)
            batch_indices = batch_indices[:, topk_indices]
        else:
            topk_batch_values = modified_values
            topk_batch_indices = batch_selected_indices
            if batch_indices is None:
                batch_indices = batch_selected_indices
            else:
                batch_indices = torch.cat((batch_indices, batch_selected_indices), dim=1)
            if batch_values is None:
                batch_values = topk_batch_values
            else:
                batch_values = torch.cat((batch_values, topk_batch_values), dim=0)
            batch_values, topk_indices = batch_values.topk(k=k_threshold)
            batch_indices = batch_indices[:, topk_indices]

        del modified_values, batch_selected_indices, topk_indices, topk_batch_values
    #   batch_size += 4000


    final_topk_values, topk_indices = batch_values.topk(k=k_threshold)
    final_topk_indices = batch_indices[:, topk_indices]

    topk_sparse_matrix = torch.sparse_coo_tensor(
        indices=final_topk_indices,
        values=final_topk_values,
        size=(X_transposed.size()[0], X_transposed.size()[0]),
        device=device
    )

    topk_sparse_matrix = topk_sparse_matrix.coalesce()

    edge_pruning_end= time.time()
    print(f"Edge Pruning finished in: {edge_pruning_end - edge_pruning_start} seconds")

    print("Length of candidates: ", len(topk_sparse_matrix.values()))

# #### check with g-t for test purposes ####
    indices_pairs = topk_sparse_matrix.indices().t().tolist()
    sorted_indices_pairs = [sorted(pair, reverse=True) for pair in indices_pairs]
#     gt = pd.read_csv("/media/wd-hdd/panpan/queryER/data/ground_truth_publications.csv", delimiter=';')
#     gt_pairs = []
#     for index, row in gt.iterrows():
#         id_d = row['id_d']
#         id_s = row['id_s']
#         gt_pairs.append([id_d, id_s])
#     indices_set = set(map(tuple, map(lambda x: map(int, x), sorted_indices_pairs)))
#     gt_set = set(map(tuple, map(lambda x: map(int, x), gt_pairs)))
#     in_ground_truth = [pair for pair in indices_set if pair in gt_set]
#     print("Pairs in Ground Truth:")
#     print(len(in_ground_truth))

    test_df = pd.DataFrame(sorted_indices_pairs, columns=['id1', 'id2'])

    test_df.to_csv(candidates_file, index=False)
    print(f"Saved candidates to: {candidates_file}")
    # d_new = dict_df.set_index('id').T.to_dict('list')
    # print(type(d_new))

    # # Create 2 lists each having IDs
    # pairs1 = pairs_df[pairs_df.columns[0]]
    # pairs2 = pairs_df[pairs_df.columns[1]]

    # # Replace the IDs with strings using the dictionary (fast method)
    # rec1 = [' '.join(d_new.get(x, None)[1:]) for x in pairs1]
    # rec2 = [' '.join(d_new.get(x, None)[1:]) for x in pairs2]

    # # In case something is None we should know, because it would mean queryER did something weird
    # for item in rec2:
    #     if(item is None):
    #         print("FOUND ONE")

    # test_df['id1'] = rec1
    # test_df['id2'] = rec2

    test_df = test_df.dropna()
    

    
    if os.path.exists(MODEL_NAME) and False:
        # Execute the code block only if the model file exists
        test_df = test_df.dropna()

        # Load the model and tokenizer
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
        tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)

        model.to(device)
        pairs_df = test_df

        # Put dummy labels for now - they shouldn't be used since it's inference
        test_df = test_df.assign(label=0)

        test_texts = test_df[['id1', 'id2']].values
        test_labels = test_df['label'].values

        print(test_labels.shape)

        # Encode the input data, this takes some time
        test_encoded_dict = [tokenizer.encode_plus(text=text1, text_pair=text2, max_length=256, padding='max_length')
                            for text1, text2 in test_texts]

        print('Sentences encoded')

        test_input_ids = [d['input_ids'] for d in test_encoded_dict]
        test_attention_masks = [d['attention_mask'] for d in test_encoded_dict]

        test_input_ids = torch.tensor(test_input_ids)
        test_attention_masks = torch.tensor(test_attention_masks)
        test_labels = torch.tensor(test_labels)

        test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

        model.eval()
        test_preds = []
        i = 1
        total_batches = int(len(test_texts) / BATCH_SIZE)

        # Begin inference, this takes more time
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_attention_masks)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            test_preds.append(logits)
            i += 1

        print("Inference done")

        test_preds = [item for sublist in test_preds for item in sublist]
        test_preds = np.argmax(test_preds, axis=1).flatten()

        print("Appending results to list")

        # Store results and convert them into arrow format to return them
        id1s = []
        id2s = []
        for i in range(0, len(test_preds)):
            if test_preds[i] == 1:
                id1s.append(pairs_df.at[i, 'id1'])
                id2s.append(pairs_df.at[i, 'id2'])

        table = pyarrow.table([
            pyarrow.array(id1s),
            pyarrow.array(id2s),
        ], names=["id1", "id2"])
        return table
    else:
        print("Model file does not exist. Skipping inference.")
        return pyarrow.table([
        pyarrow.array([], type=pyarrow.uint32()),
        pyarrow.array([], type=pyarrow.uint32()),
        ], names=["id1", "id2"])
