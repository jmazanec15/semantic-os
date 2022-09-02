"""

"""
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25


def generate_bm25_result(index_name, host_name, corpus, queries, initialize=False, number_of_shards=1,
                         k_values=[1,3,5,10,100,1000]):
    '''
         This method generates the search score of the given questions and pulls answers from elastic search
         We create an index in elastic search and then ingest the documents and then perform search based on queries

         @param index_name:        string    name of the index

         @param host_name:         string    host name to connect to the elastic search

         @param corpus:            list      a list of dictionaries with document id and title

         @param queries:           list      a list of dictionaries with question id and title and metadata

         @param initialize:        boolean   if initialize is true then we will create a index and ingest documents to
                                                the index, otherwise it will just perform the search in the existing
                                                index

         @param number_of_shards   integer   number of shards in the elastic search index

         @return results           dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score, search result of bm25
         @return retriever         object    class object of Retriever

    '''


    # data folder would contain these files:
    # (1) scifact/corpus.jsonl  (format: jsonlines)
    # (2) scifact/queries.jsonl (format: jsonlines)
    # (3) scifact/qrels/test.tsv (format: tsv ("\t"))

    #### Lexical Retrieval using Bm25 (Elasticsearch) ####
    #### Provide a hostname (localhost) to connect to ES instance
    #### Define a new index name or use an already existing one.
    #### We use default ES settings for retrieval
    #### https://www.elastic.co/
    model = BM25(index_name=index_name, hostname=host_name, initialize=initialize, number_of_shards=number_of_shards)

    retriever = EvaluateRetrieval(model, k_values=k_values)
    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    return results, retriever
