## Reranker IR Benchmark (as of 2024. 12. 30.)
* Note that cross-encoder only with AutoRAG dataset is applied at this moment.
* The target language is Korean, and the task is IR (Reranking).

| **Model Name**                     | **Accuracy@1** | **Accuracy@3** | **Accuracy@5** | **Accuracy@10** | **F1@1** | **F1@3** | **F1@5** | **F1@10** | **Inference Time (s)** | **Avg Inference Time/query (s)** |
|------------------------------------|----------------|----------------|----------------|-----------------|----------|----------|----------|-----------|-----------------------|----------------------------------|
| dragonkue/bge-reranker-v2-m3-ko   | 0.912          | 0.965          | 0.965          | 0.974           | 0.912    | 0.482    | 0.322    | 0.177     | 310.29               | 2.72                             |
| sigridjineth/ko-reranker-v1.1     | 0.807          | 0.921          | 0.947          | 0.974           | 0.807    | 0.461    | 0.316    | 0.177     | 142.64               | 1.25                             |
| Alibaba-NLP/gte-multilingual-reranker-base | 0.728    | 0.921          | 0.947          | 0.974           | 0.728    | 0.461    | 0.316    | 0.177     | 252.35               | 2.21                             |
| sigridjineth/ko-reranker-v1.2-preview | 0.877       | 0.947          | 0.965          | 0.974           | 0.877    | 0.474    | 0.322    | 0.177     | 219.37               | 1.92                             |
| upskyy/ko-reranker-8k             | 0.868          | 0.956          | 0.965          | 0.982           | 0.868    | 0.478    | 0.322    | 0.179     | 338.31               | 2.97                             |
