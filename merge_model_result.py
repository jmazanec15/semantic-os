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


def normalize_values(seach_result):
    '''
         This method normalizes values of a nested dictionary with in-place replacement.

         @param seach_result:        dict     a nested dictionary of question id and dictionary of document id and score

         @return seach_result:       dict     a nested dictionary of question id and dictionary of document id and
                                              normalized score
    '''
    for _, d in seach_result.items():
        x_array = np.array(list(d.values())).astype(float)
        normalized_arr = preprocessing.normalize([x_array])
        normalized_arr = normalized_arr.tolist()[0]

        # total_sum = sum(d.values())
        # factor = 1.0 / total_sum
        d.update(zip(d, normalized_arr))

    return seach_result


def get_mean_result(bm25_result, sbert_result, meanType="harmonic"):
    '''
         This method calculates the mean of the search result from bm25 and bert model

         @param bm25_result:        dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score
                                              search result of bm25

         @param sbert_result:      dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score
                                              search result of bert transformer

         @param final_result:      dict      a nested dictionary of question id and dictionary of document id and
                                              mean score of both results

    '''
    print(meanType + " is calculating")

    final_result = defaultdict()
    for question_id, doc_dict in sbert_result.items():
        for doc_id, doc_value in doc_dict.items():
            if question_id in bm25_result.keys() and doc_id in bm25_result[question_id].keys():
                if question_id not in final_result.keys():
                    final_result[question_id] = {}
                if meanType == "arithmatic":
                    final_result[question_id][doc_id] = mean([bm25_result[question_id][doc_id],
                                                              sbert_result[question_id][doc_id]])
                elif meanType == "geometric":
                    final_result[question_id][doc_id] = geometric_mean([bm25_result[question_id][doc_id],
                                                                        sbert_result[question_id][doc_id]])
                else:
                    final_result[question_id][doc_id] = harmonic_mean([bm25_result[question_id][doc_id],
                                                                       sbert_result[question_id][doc_id]])

    return final_result


def get_normalized_weighted_linear_result(bm25_result, sbert_result, factor=1.0):
    '''
         This method calculates the mean of the search result from bm25 and bert model
         score = first_result * factor + second_result

         @param bm25_result:        dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score, search result of bm25

         @param sbert_result:      dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score, search result of bert transformer

         @param factor:            number    weight factor to be added with sbert result

         @return final_result:     dict      a nested dictionary of question id and dictionary of document id and
                                              weighted linear result

    '''
    print("factor: ", factor)

    final_result = defaultdict()
    for question_id, doc_dict in sbert_result.items():
        for doc_id, doc_value in doc_dict.items():
            if question_id in bm25_result.keys() and doc_id in bm25_result[question_id].keys():
                if question_id not in final_result.keys():
                    final_result[question_id] = {}
                if doc_id in bm25_result[question_id].keys():
                    final_result[question_id][doc_id] = bm25_result[question_id][doc_id] + \
                                                        (factor * sbert_result[question_id][doc_id])
                else:
                    final_result[question_id][doc_id] = 0 + (factor * sbert_result[question_id][doc_id])

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

tt_k_values = [1, 3, 5, 10, 100, len(corpus)]

fh_k_values = [1, 3, 5, 10, 100, 250]

k_values = [1, 3, 5, 10, 100]

index_name = dataset
host_name = "localhost"

bm25_result, bm25_retriever = generate_bm25_result(index_name, host_name, corpus, queries, initialize=False,
                                                   k_values=tt_k_values)

count_bm25_dict = count_output_list(bm25_result)

# print("number of the question that doesn't have 100 BM25 results: ", len(count_bm25_dict))
#
# print(count_bm25_dict)

bm25_norm_result = normalize_values(bm25_result)

## this is for custom model. In `generate_sbert_result` you can either provide the model name or the file path of the
# model
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/custom_tasb"

# this section takes a lot of time as this generates embedding for questions and answers and then finds the
# similary between both embedded values
# sbert_result, dense_retriever = generate_sbert_result(corpus, queries, custom_model_path, fh_k_values,
#                                                       batch_size=16)
sbert_result, dense_retriever = generate_sbert_result(corpus, queries, "msmarco-roberta-base-ance-firstp", fh_k_values,
                                                      batch_size=16)

count_dense_dict = count_output_list(sbert_result)

# print("number of the question that doesn't have 100 DenseModel results: ", len(count_dense_dict))

sbert_norm_result = normalize_values(sbert_result)

# merged_result = get_mean_result(bm25_norm_result, sbert_norm_result, meanType="arithmatic")
merged_result = get_normalized_weighted_linear_result(bm25_norm_result, sbert_norm_result, 2)

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
