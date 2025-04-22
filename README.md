# Make Reranker Benchmark Simple Again
## Purpose
* 본 프로젝트의 목표는 Reranker Benchmark Evaluation을 최소한의 의존성으로 경량화하여, 쉽게 실행하고 즉각적인 결과를 얻을 수 있도록 설계하는 것입니다.

## Results
* 본 프로젝트에서는 BM25 기반의 1차 Retrieval을 통해 각 벤치마크 query 당 retrieval corpus를 1000개로 제한합니다.
* 이후 1000개 corpus 내에서 Reranker 모델의 성능을 평가합니다.

### Stage 1
* 최적의 성능을 보여주는 한국어 tokenizer를 선정하기 위해 tokenizer별 평가를 진행하였습니다. 관련 코드는 이곳을 참고해 주세요.
* 평가 Tokenizer 목록: Kiwi, Kkma, Mecab, Okt
* 평가 데이터셋: AutoRAGRetrieval, BelebeleRetrieval, Ko-StrategyQA, PublicHealthQA
* 평가 결과는 아래 표와 같습니다.

#### Top-k 10
| Model | Average Recall | Average Precision | Average NDCG | Average F1 |
|-------|----------------|-------------------|--------------|------------|
| Mecab | **0.8731**     | 0.1000            | 0.7433       | **0.1783** |
| Okt   | 0.8655         | **0.1001**        | **0.7474**   | 0.1783     |
| Kkma  | 0.8504         | 0.0982            | 0.7358       | 0.1749     |
| Kiwi  | 0.8443         | 0.0961            | 0.7210       | 0.1715     |

#### Top-k 100
| Model | Average Recall | Average Precision | Average NDCG | Average F1 |
|-------|----------------|-------------------|--------------|------------|
| Okt   | **0.9534**     | **0.0113**        | **0.7688**   | **0.0223** |
| Mecab | 0.9533         | **0.0113**        | 0.7634       | **0.0223** |
| Kkma  | 0.9528         | **0.0113**        | 0.7616       | **0.0223** |
| Kiwi  | 0.9454         | 0.0111            | 0.7464       | 0.0220     |

#### Top-k 1000
| Model | Average Recall | Average Precision | Average NDCG | Average F1 |
|-------|----------------|-------------------|--------------|------------|
| Kkma  | **0.9743**     | **0.0012**        | 0.7648       | **0.0023** |
| Mecab | 0.9711         | 0.0012            | 0.7661       | 0.0023     |
| Okt   | 0.9688         | 0.0012            | **0.7712**   | 0.0023     |
| Kiwi  | 0.9671         | 0.0012            | 0.7497       | 0.0023     |

<!-- ## Dataset
* The target language is Korean at this moment.
* [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark/pull/6) (DATATYPE_NAME=AutoRAG)
* (planned, not yet) [KURE](https://github.com/nlpai-lab/KURE)

## Models
* HuggingFace Reranker `MODEL_CLASS=huggingface`
* FlagReranker `MODEL_CLASS=flagreranker`
  * e.g. `BAAI/bge-reranker-v2-m3`
* FlagLLMReranker `MODEL_CLASS=flagllmreranker`
  * e.g. `BAAI/bge-reranker-v2-gemma`
* FlagLayerwiseReranker `MODEL_CLASS=flaglayerwise`
  * e.g. `BAAI/bge-reranker-v2.5-gemma2-lightweight`
* (planned, not yet) HuggingFace & FlagEmbedding supported bi-encoder

## Command
### Setup
```
make init
```

### Run
```
# single GPU only at the moment.
make run TYPE=cross-encoder MODEL_NAME=sigridjineth/ko-reranker-v1.1 MODEL_CLASS=huggingface DATATYPE_NAME=AutoRAG
make run TYPE=cross-encoder MODEL_NAME=BAAI/bge-reranker-v2-m3 MODEL_CLASS=flagreranker DATATYPE_NAME=AutoRAG
make run MODEL_NAME=BAAI/bge-reranker-v2-gemma MODEL_CLASS=flagllmreranker
```

## Contributions

This project welcomes contributions and suggestions. See [issues](https://github.com/instructkr/retriever-simple-benchmark/issues) if you consider doing any.

When you submit a pull request, please make sure that you should run formatter by `make format && make check`, please. -->