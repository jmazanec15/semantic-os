from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


def generate_sbert_result(corpus, queries, model_name, k_values, batch_size=16):

    '''
         This method generates the similarity score of the givern questions and queries based on the model name provided

         @param corpus:            list      a list of dictionaries with document id and title

         @param queries:           list      a list of dictionaries with question id and title and metadata

         @param model_name:        string    name of the sbert model to generate the semantic score

         @param k_values           list      a list of integers, max of the list is defined how many similarities
                                                will be found
         @param batch_size         int       size of the batch to generate embedding of the documents

         @return results           dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score, search result of bm25
         @return retriever         object    class object of Retriever

    '''

    #### Load the SBERT model and retrieve using cosine-similarity
    model = DRES(models.SentenceBERT(model_name), batch_size=batch_size)

    # dataset = "nfcorpus"
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    # data_path = util.download_and_unzip(url, out_dir)
    #
    # corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    retriever = EvaluateRetrieval(model, score_function="cos_sim",
                                  k_values=k_values)  # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)

    return results, retriever
