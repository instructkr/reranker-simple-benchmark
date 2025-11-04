# Make Reranker Benchmark Simple Again
## Purpose
* 본 프로젝트는 Reranker Benchmark Evaluation을 최소한의 의존성으로 경량화하여, 누구나 쉽게 실행하고 즉각적인 결과를 얻을 수 있도록 설계되었습니다.

## Plan
* 본 프로젝트에서는 BM25 기반의 Stage 1 Retrieval을 통해 각 벤치마크 query 당 retrieval corpus를 1000개로 제한합니다. 각 query에 대한 정답 문서 정보를 포함하여, BM25 기준 상위 1000개 문서의 ID를 저장합니다.
* 이후 각 query 당 Top-k 50개의 corpus id를 활용하여, Stage 2 Reranking을 진행합니다.

## Results
### Stage 1 Retrieval
최적의 성능을 보여주는 한국어 tokenizer를 선정하기 위해 tokenizer별 평가를 진행하였습니다. 
#### Evaluation Code
```bash
uv run retrieve_bm25_with_tokenize.py \
	--tokenizer_list all \
	--data_list all
```
#### Leaderboard
```bash
cd eval
uv run streamlit run leaderboard_bm25.py
```
#### Results
| Model | Average Recall@10 | Average Precision@10 | Average NDCG@10 | Average F1@10 |
|-------|----------------|-------------------|--------------|------------|
| Mecab | **0.8731**     | 0.1000            | 0.7433       | **0.1783** |
| Okt   | 0.8655         | **0.1001**        | **0.7474**   | 0.1783     |
| Kkma  | 0.8504         | 0.0982            | 0.7358       | 0.1749     |
| Kiwi  | 0.8443         | 0.0961            | 0.7210       | 0.1715     |

top-k 10에서 가장 높은 성능을 보인 **Mecab** tokenizer를 사용하여, Stage 1 Retrieval을 진행하였습니다.

### Stage 2 Reranking
#### Benchmark Datasets
[MTEB](https://github.com/embeddings-benchmark/mteb)에 등록된 모든 Korean Retrieval Benchmark에 대한 평가를 진행하였습니다.
- [Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA): 한국어 ODQA multi-hop 검색 데이터셋 (StrategyQA 번역)
- [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm): 금융, 공공, 의료, 법률, 커머스 5개 분야에 대해, pdf를 파싱하여 구성한 한국어 문서 검색 데이터셋
- [MIRACLRetrieval](https://huggingface.co/datasets/miracl/miracl): Wikipedia 기반의 한국어 문서 검색 데이터셋
- [PublicHealthQA](https://huggingface.co/datasets/xhluca/publichealth-qa): 의료 및 공중보건 도메인에 대한 한국어 문서 검색 데이터셋
- [BelebeleRetrieval](https://huggingface.co/datasets/facebook/belebele): FLORES-200 기반의 한국어 문서 검색 데이터셋
- [MrTidyRetrieval](https://huggingface.co/datasets/mteb/mrtidy): Wikipedia 기반의 한국어 문서 검색 데이터셋
- [MultiLongDocRetrieval](https://huggingface.co/datasets/Shitao/MLDR): 다양한 도메인의 한국어 장문 검색 데이터셋
- [XPQARetrieval](https://huggingface.co/datasets/jinaai/xpqa): 다양한 도메인의 한국어 문서 검색 데이터셋

#### Evaluation Code
```bash
cd eval
uv run evaluate_reranker.py \
	--model_names "my_reranker_model" \
	--tasks "Ko-StrategyQA AutoRAGRetrieval MIRACLRetrieval PublicHealthQA BelebeleRetrieval MrTidyRetrieval MultiLongDocRetrieval XPQARetrieval" \
	--gpu_ids 0 1 2 3 4 5 6 7 \
	--previous_results_dir "results/stage1" \
	--output_dir "results/stage2" \
	--top_k 50 \
	--verbosity 1 \
	--batch_size 2
```

#### Leaderboard
```bash
cd eval
uv run streamlit run leaderboard_reranker.py
```

#### Results
| Model                                  | Average MRR@1 | Average MAP@1 | Average NDCG@1 |
|----------------------------------------|---------------|---------------|----------------|
| Qwen3-Reranker-4B-seq-cls              | 0.7597        | 0.6036        | 0.6972         |
| Qwen3-Reranker-8B-seq-cls              | 0.7488        | 0.6268        | 0.7253         |
| bge-reranker-v2-m3                     | 0.7298        | 0.6054        | 0.6967         |
| Qwen3-Reranker-0.6B-seq-cls            | 0.7278        | 0.5167        | 0.5937         |
| mxbai-rerank-large-v2                  | 0.7119        | 0.6145        | 0.7083         |
| PIXIE-Spell-Reranker-Preview-0.6B      | 0.7062        | 0.6100        | 0.6971         |
| bge-reranker-v2-m3-ko                  | 0.7043        | 0.5689        | 0.6512         |
| ko-reranker-8k                         | 0.7039        | 0.5289        | 0.6128         |
| bge-reranker-v2-gemma                  | 0.7035        | 0.5932        | 0.7011         |
| gte-multilingual-reranker-base         | 0.6904        | 0.5872        | 0.6791         |
| ko-reranker-v1.1                       | 0.6696        | 0.5790        | 0.6662         |
| ko-reranker                            | 0.6412        | 0.5113        | 0.5992         |
| jina-reranker-v2-base-multilingual     | 0.6272        | 0.5341        | 0.6285         |
| jina-reranker-v3                       | 0.6128        | 0.527         | 0.6129         |

| Model                                  | Average MRR@5 | Average MAP@5 | Average NDCG@5 |
|----------------------------------------|---------------|---------------|----------------|
| Qwen3-Reranker-4B-seq-cls              | 0.7991        | 0.7423        | 0.7757         |
| Qwen3-Reranker-8B-seq-cls              | 0.7964        | 0.7578        | 0.7898         |
| bge-reranker-v2-m3                     | 0.7777        | 0.7339        | 0.7669         |
| Qwen3-Reranker-0.6B-seq-cls            | 0.7698        | 0.6601        | 0.7025         |
| mxbai-rerank-large-v2                  | 0.7692        | 0.7363        | 0.7691         |
| bge-reranker-v2-gemma                  | 0.7622        | 0.7304        | 0.7646         |
| PIXIE-Spell-Reranker-Preview-0.6B      | 0.7610        | 0.7275        | 0.7594         |
| gte-multilingual-reranker-base         | 0.7548        | 0.7156        | 0.7519         |
| bge-reranker-v2-m3-ko                  | 0.7543        | 0.6935        | 0.7290         |
| ko-reranker-8k                         | 0.7486        | 0.6655        | 0.7050         |
| ko-reranker-v1.1                       | 0.7306        | 0.6957        | 0.7310         |
| ko-reranker                            | 0.6945        | 0.6459        | 0.6817         |
| jina-reranker-v2-base-multilingual     | 0.6888        | 0.6585        | 0.6919         |
| jina-reranker-v3                       | 0.6815        | 0.6309        | 0.6712         |

| Model                                  | Average MRR@10 | Average MAP@10 | Average NDCG@10 |
|----------------------------------------|----------------|----------------|-----------------|
| Qwen3-Reranker-4B-seq-cls              | 0.8039         | 0.7547         | 0.7926          |
| Qwen3-Reranker-8B-seq-cls              | 0.8000         | 0.7698         | 0.8044          |
| bge-reranker-v2-m3                     | 0.7821         | 0.7462         | 0.7832          |
| mxbai-rerank-large-v2                  | 0.7752         | 0.7509         | 0.7901          |
| Qwen3-Reranker-0.6B-seq-cls            | 0.7740         | 0.6754         | 0.7247          |
| bge-reranker-v2-gemma                  | 0.7673         | 0.7450         | 0.7837          |
| PIXIE-Spell-Reranker-Preview-0.6B      | 0.7661         | 0.7405         | 0.7775          |
| bge-reranker-v2-m3-ko                  | 0.7605         | 0.7074         | 0.7509          |
| gte-multilingual-reranker-base         | 0.7598         | 0.7288         | 0.7705          |
| ko-reranker-8k                         | 0.7544         | 0.6795         | 0.7267          |
| ko-reranker-v1.1                       | 0.7361         | 0.7094         | 0.7508          |
| ko-reranker                            | 0.7021         | 0.6610         | 0.7055          |
| jina-reranker-v2-base-multilingual     | 0.6956         | 0.6725         | 0.7129          |
| jina-reranker-v3                       | 0.6902         | 0.6481         | 0.7002          |

<!-- ## Contributions

This project welcomes contributions and suggestions. See [issues](https://github.com/instructkr/retriever-simple-benchmark/issues) if you consider doing any.

When you submit a pull request, please make sure that you should run formatter by `make format && make check`, please. -->