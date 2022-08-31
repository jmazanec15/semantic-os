from beir import util, LoggingHandler
import logging
import pathlib, os, random
from bm25_result import generate_bm25_result
from sbert_result import generate_sbert_result
from beir.datasets.data_loader import GenericDataLoader
from statistics import harmonic_mean, geometric_mean
from sklearn import preprocessing
import numpy as np
from collections import defaultdict


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

def get_mean_result(bm25_result, sbert_result, meanType = "harmonic"):
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
    final_result = defaultdict()
    for question_id, doc_dict in sbert_result.items():
        for doc_id, doc_value in doc_dict.items():
            if question_id in bm25_result.keys() and doc_id in bm25_result[question_id].keys():
                if question_id not in final_result.keys():
                    final_result[question_id] = {}
                if meanType == "harmonic":
                    final_result[question_id][doc_id] = harmonic_mean([bm25_result[question_id][doc_id],
                                                                       sbert_result[question_id][doc_id]])

    return final_result



k_values = [1, 3, 5, 10, 100]

#### /print debug information to stdout

#### Download dataset and unzip the dataset
dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

index_name = dataset
host_name = "localhost"

bm25_result, bm25_retriever = generate_bm25_result(index_name, host_name, corpus, queries)

bm25_norm_result = normalize_values(bm25_result)

# this section takes a lot of time as this generates embedding for questions and answers and then finds the
# similary between both embedded values
sbert_result, dense_retriever = generate_sbert_result(corpus, queries, "msmarco-distilbert-base-tas-b", k_values, 128)
#
sbert_norm_result = normalize_values(sbert_result)

merged_result = get_mean_result(bm25_norm_result, sbert_norm_result)
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, merged_result, k_values)

print("Printing ndcg:", ndcg)
print("Printing _map:", _map)
print("Printing recall:", recall)
print("Printing precision:", precision)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(merged_result.items()))
logging.info("Query : %s\n" % queries[query_id])

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    doc_id = scores[rank][0]
    logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
