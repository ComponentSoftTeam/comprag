# RAG pipeline

A similar project's study: https://arxiv.org/html/2401.01511v1

## Hyperparameters
- $N$: Number of embedding functions
- $M$: Number of search type-s (like BM25, TF-IDF, etc.)

## Databases
- Local database: Chroma
- $N$ different databases, each of each time of embeddings
- 1 additional for plane text

## Telemetry / Metrics
https://developer.nvidia.com/blog/evaluating-retriever-for-enterprise-grade-rag/
For each method we want to collect the following telemetry:
- How many times did the retrieved result ended up in the final prompt
- Total number of retrieved results
- Number of relevant results
- Recall (nvidia)
- NDCG (nvidia)
- https://medium.com/llamaindex-blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83

## Retrievers
https://arxiv.org/html/2404.07220v1
- We also experiment with different retriever strategies
- Like query expansion, etc.

## Rerankers

https://medium.com/llamaindex-blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83
- bge-m3


## RAG
- A given query is run for
    - every database for every search-type that uses embedding
    - every search-type that uses plane text
- In the end, there are a total of $\Theta(N \times M + M)$ results 
- For each search, there is a number of retrived documents, we will flatten this to a single list of documents
- Note that in a perfect world there are a lot of duplicates, so we will deduplicate the answers
- We want to collect telemetry about the "goodness" of each method we use, to we will flattan and deduplicate while retaining metadata
- For each result we will compute a relevance to the original query (with the renranker)
- For each pair of results we will compute their similarity (With the reranker)
- With the relevance and the similarity matrix we will get the top K most relevant results with the most information
    - This is know as the maximum coverage problem (with the similarity as a penlalty)


## RAG benchmark
The rag will be benchmarked according to the techniques described in arXiv:2309.01431(cs)


