import logging
import os
import pathlib
import random
from collections import defaultdict
from statistics import harmonic_mean, mean, geometric_mean

import numpy as np
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from sklearn import preprocessing

from bm25_result import generate_bm25_result
from sbert_result import generate_sbert_result

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def count_output_list(merged_result):
    count_dict = {}
    for question_id, doc_dict in merged_result.items():
        if len(doc_dict) < 100:
            count_dict[question_id] = len(doc_dict)
    return count_dict


def rerank_on_subset(superset_results, rerank_results, subset_size):
    """
    Reranks subset of results from one set based on order from other set

    @param superset_results: Results that will produce subset to be reranked
    @param rerank_results: Results to rerank based on
    @param subset_size: size of subset
    @return: reranked results
    """
    final_result = defaultdict()
    for question_id, doc_dict in superset_results.items():
        # First, lets get the top subset_size docs
        q_subset_size = min(len(doc_dict.keys()), subset_size)
        new_results = dict(sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)[0:q_subset_size])
        print("New Results: {}".format(new_results))
        for key in new_results.keys():
            if question_id not in rerank_results or key not in rerank_results[question_id]:
                new_results[key] = 0
            else:
                new_results[key] = rerank_results[question_id][key]

        final_result[question_id] = new_results

    return final_result



#### /print debug information to stdout

#### Download dataset and unzip the dataset
dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# This k values are being used for BM25 search
subset_size = 500
tt_k_values = [subset_size]

# This K values are being used for dense model search
fh_k_values = [len(corpus)]

# this k values are being used for scoring
k_values = [1, 3, 5, 10, 100]

index_name = dataset
host_name = "localhost"

bm25_result, bm25_retriever = generate_bm25_result(index_name, host_name, corpus, queries, initialize=False,
                                                   k_values=tt_k_values)

count_bm25_dict = count_output_list(bm25_result)
print("number of the question that doesn't have 100 BM25 results: ", len(count_bm25_dict))

## this is for custom model. In `generate_sbert_result` you can either provide the model name or the file path of the
# model
custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/custom_tasb"

# this section takes a lot of time as this generates embedding for questions and answers and then finds the
# similary between both embedded values
sbert_result, dense_retriever = generate_sbert_result(corpus, queries, custom_model_path, fh_k_values,
                                                      batch_size=16)

count_dense_dict = count_output_list(sbert_result)

print("number of the question that doesn't have 100 DenseModel results: ", len(count_dense_dict))

merged_result = rerank_on_subset(bm25_result, sbert_result, subset_size)

print("Number of questions:", len(merged_result))

count_dict = count_output_list(merged_result)

print("number of the question that doesn't have 100 results: ", len(count_dict))

ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, merged_result, k_values)

print("Printing ndcg:", ndcg)
print("Printing _map:", _map)
print("Printing precision:", precision)
print("Printing recall:", recall)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(sbert_result.items()))
logging.info("Query : %s\n" % queries[query_id])

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    doc_id = scores[rank][0]
    logging.info("Doc %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
