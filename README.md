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
**10개의 Korean Retrieval Benchmark** (총 18,945 queries)에 대한 평가를 진행하였습니다.

**MTEB 데이터셋 (8개)**:
- [Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA): 한국어 ODQA multi-hop 검색 데이터셋 (StrategyQA 번역) - 592 queries
- [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm): 금융, 공공, 의료, 법률, 커머스 5개 분야 한국어 문서 검색 - 114 queries
- [MIRACLRetrieval](https://huggingface.co/datasets/miracl/miracl): Wikipedia 기반 한국어 문서 검색 - 213 queries
- [PublicHealthQA](https://huggingface.co/datasets/xhluca/publichealth-qa): 의료 및 공중보건 도메인 한국어 문서 검색 - 77 queries
- [BelebeleRetrieval](https://huggingface.co/datasets/facebook/belebele): FLORES-200 기반 한국어 문서 검색 - 900 queries
- [MrTidyRetrieval](https://huggingface.co/datasets/mteb/mrtidy): Wikipedia 기반 한국어 문서 검색 - 421 queries
- [MultiLongDocRetrieval](https://huggingface.co/datasets/Shitao/MLDR): 다양한 도메인 한국어 장문 검색 - 200 queries
- [XPQARetrieval](https://huggingface.co/datasets/jinaai/xpqa): 다양한 도메인 한국어 문서 검색 - 654 queries

**커스텀 데이터셋 (2개)**:
- [SQuADKorV1Retrieval](https://huggingface.co/datasets/yjoonjang/squad_kor_v1): 한국어 SQuAD v1.0 기반 검색 - 5,774 queries
- [WebFAQRetrieval](https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval): 한국어 웹 FAQ 검색 - 10,000 queries

> **Note**: 커스텀 데이터셋은 `eval/custom_mteb_tasks.py`에 MTEB Task 클래스로 구현되어 있습니다.

#### Evaluation Code
```bash
# 모든 10개 데이터셋 평가 (DEFAULT_TASKS)
uv run python eval/evaluate_reranker.py \
	--model_names BAAI/bge-reranker-v2-m3 \
	--gpu_ids 0 1 2 3 4 5 6 7 \
	--batch_size 2 \
	--top_k 50 \
	--verbosity 1

# 또는 특정 데이터셋만 선택
uv run python eval/evaluate_reranker.py \
	--model_names "my_reranker_model" \
	--tasks Ko-StrategyQA AutoRAGRetrieval SQuADKorV1Retrieval WebFAQRetrieval \
	--gpu_ids 0 1 2 3 \
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
| Qwen3-Reranker-4B-seq-cls              | 0.7926        | 0.6517        | 0.7266         |
| Qwen3-Reranker-8B-seq-cls              | 0.7810        | 0.6761        | 0.7549         |
| Qwen3-Reranker-0.6B-seq-cls            | 0.7698        | 0.5583        | 0.6200         |
| bge-reranker-v2-m3                     | 0.7632        | 0.6608        | 0.7338         |
| ko-reranker-8k                         | 0.7469        | 0.5927        | 0.6598         |
| bge-reranker-v2-m3-ko                  | 0.7462        | 0.6288        | 0.6946         |
| mxbai-rerank-large-v2                  | 0.7457        | 0.6670        | 0.7420         |
| PIXIE-Spell-Reranker-Preview-0.6B      | 0.7424        | 0.6633        | 0.7331         |
| gte-multilingual-reranker-base         | 0.7158        | 0.6319        | 0.7054         |
| ko-reranker-v1.1                       | 0.7094        | 0.6357        | 0.7054         |
| jina-reranker-v3                       | 0.7013        | 0.6277        | 0.7013         |
| bge-reranker-v2-gemma                  | 0.6953        | 0.6176        | 0.6931         |
| ko-reranker                            | 0.6924        | 0.5823        | 0.6525         |
| jina-reranker-v2-base-multilingual     | 0.6717        | 0.5966        | 0.6721         |

| Model                                  | Average MRR@5 | Average MAP@5 | Average NDCG@5 |
|----------------------------------------|---------------|---------------|----------------|
| Qwen3-Reranker-4B-seq-cls              | 0.8283        | 0.7732        | 0.8037         |
| Qwen3-Reranker-8B-seq-cls              | 0.8243        | 0.7892        | 0.8176         |
| bge-reranker-v2-m3                     | 0.8075        | 0.7707        | 0.7998         |
| Qwen3-Reranker-0.6B-seq-cls            | 0.8059        | 0.6903        | 0.7305         |
| mxbai-rerank-large-v2                  | 0.7993        | 0.7724        | 0.8014         |
| PIXIE-Spell-Reranker-Preview-0.6B      | 0.7926        | 0.7645        | 0.7927         |
| bge-reranker-v2-m3-ko                  | 0.7906        | 0.7367        | 0.7680         |
| ko-reranker-8k                         | 0.7869        | 0.7119        | 0.7470         |
| gte-multilingual-reranker-base         | 0.7765        | 0.7443        | 0.7769         |
| ko-reranker-v1.1                       | 0.7653        | 0.7367        | 0.7677         |
| jina-reranker-v3                       | 0.7615        | 0.7356        | 0.7680         |
| bge-reranker-v2-gemma                  | 0.7494        | 0.7268        | 0.7561         |
| ko-reranker                            | 0.7407        | 0.6984        | 0.7300         |
| jina-reranker-v2-base-multilingual     | 0.7289        | 0.7042        | 0.7339         |

| Model                                  | Average MRR@10 | Average MAP@10 | Average NDCG@10 |
|----------------------------------------|----------------|----------------|-----------------|
| Qwen3-Reranker-4B-seq-cls              | 0.8324         | 0.7836         | 0.8181          |
| Qwen3-Reranker-8B-seq-cls              | 0.8275         | 0.7991         | 0.8302          |
| bge-reranker-v2-m3                     | 0.8113         | 0.7810         | 0.8137          |
| Qwen3-Reranker-0.6B-seq-cls            | 0.8095         | 0.7034         | 0.7503          |
| mxbai-rerank-large-v2                  | 0.8044         | 0.7845         | 0.8191          |
| PIXIE-Spell-Reranker-Preview-0.6B      | 0.7971         | 0.7754         | 0.8082          |
| bge-reranker-v2-m3-ko                  | 0.7960         | 0.7482         | 0.7866          |
| ko-reranker-8k                         | 0.7919         | 0.7698         | 0.7654          |
| gte-multilingual-reranker-base         | 0.7813         | 0.7754         | 0.7937          |
| ko-reranker-v1.1                       | 0.7702         | 0.7481         | 0.7847          |
| jina-reranker-v3                       | 0.7673         | 0.7488         | 0.7886          |
| bge-reranker-v2-gemma                  | 0.7553         | 0.7394         | 0.7751          |
| ko-reranker                            | 0.7472         | 0.7110         | 0.7502          |
| jina-reranker-v2-base-multilingual     | 0.7349         | 0.7160         | 0.7521          |

<!-- ## Contributions

This project welcomes contributions and suggestions. See [issues](https://github.com/instructkr/retriever-simple-benchmark/issues) if you consider doing any.

When you submit a pull request, please make sure that you should run formatter by `make format && make check`, please. -->