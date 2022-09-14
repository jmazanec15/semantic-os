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


def combine_results(results_a, results_b, num_results):
    """
    Simple strategy to provide one result from each. Skip duplicates. If there are not enough from one set. Use the
    rest from the other set

    @param results_a: First set of results
    @param results_b: Second set of results
    @param num_results: Number of results to aggregate to
    @return: set of results
    """

    questions = [question_id for question_id, doc_dict in results_a.items()]
    questions.extend([question_id for question_id, doc_dict in results_b.items() if question_id not in questions])

    final_result = dict()
    for question_id in questions:
        final_result[question_id] = dict()

        # First, sort the results into lists so that the best results at the fron for the question
        q_results_a = results_a[question_id] if question_id in results_a else {}
        q_results_a_iterable = list(sorted(q_results_a.items(), key=lambda item: item[1], reverse=True))
        q_results_b = results_b[question_id] if question_id in results_b else {}
        q_results_b_iterable = list(sorted(q_results_b.items(), key=lambda item: item[1], reverse=True))

        size_a = len(q_results_a_iterable)
        size_b = len(q_results_b_iterable)

        a_i = 0
        b_i = 0
        current_i = 0
        a_turn = True if size_a > 0 else False

        # Go one by one through the results adding the correct next one. Skip if it has already been added
        while current_i < min(size_a + size_b, num_results):
            if a_turn:
                while a_i < size_a and q_results_a_iterable[a_i][0] in final_result[question_id]:
                    a_i += 1

                if a_i < size_a:
                    final_result[question_id][q_results_a_iterable[a_i][0]] = 1.0 / (current_i + 1)
                    current_i += 1
                a_turn = False
            else:
                while b_i < size_b and q_results_b_iterable[b_i][0] in final_result[question_id]:
                    b_i += 1

                if b_i < size_b:
                    final_result[question_id][q_results_b_iterable[b_i][0]] = 1.0 / (current_i + 1)
                    current_i += 1
                a_turn = True

    return final_result


def count_output_list(merged_result):
    count_dict = {}
    for question_id, doc_dict in merged_result.items():
        if len(doc_dict) < 100:
            count_dict[question_id] = len(doc_dict)
    return count_dict




#### /print debug information to stdout

#### Download dataset and unzip the dataset
dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# This k values are being used for BM25 search
tt_k_values = [len(corpus)]

# This K values are being used for dense model search
fh_k_values = [len(corpus)]

# this k values are being used for scoring
k_values = [1, 3, 5, 10, 100]

index_name = dataset
host_name = "localhost"

bm25_result, bm25_retriever = generate_bm25_result(index_name, host_name, corpus, queries, initialize=False,
                                                   k_values=tt_k_values)

count_bm25_dict = count_output_list(bm25_result)

custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/custom_tasb"

# this section takes a lot of time as this generates embedding for questions and answers and then finds the
# similary between both embedded values
sbert_result, dense_retriever = generate_sbert_result(corpus, queries, custom_model_path, fh_k_values,
                                                      batch_size=16)

count_dense_dict = count_output_list(sbert_result)

merged_result = combine_results(bm25_result, sbert_result, 100)

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
