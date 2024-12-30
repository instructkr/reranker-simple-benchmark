# Make Running Benchmark Simple Again
## Purpose
* The goal is to redesign retrieval/reranker benchmark evaluation projects lightweight, with minimal dependencies, that runs effortlessly and delivers immediate results.

## Results
* 벤치마크 결과는 [README.md](https://github.com/instructkr/retriever-simple-benchmark/blob/main/results/README.md) 에서 확인하세요.

## Dataset
* The target language is Korean at this moment.
* [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark/pull/6) (DATATYPE_NAME=AutoRAG)
* (planned, not yet) [KURE](https://github.com/nlpai-lab/KURE)

## Models
* HuggingFace (MODEL_CLASS=huggingface) & FlagReranker (MODEL_CLASS=flagreranker) & FlagLLMReranker (MODEL_CLASS=flagllmreranker) supported cross-encoder
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

When you submit a pull request, please make sure that you should run formatter by `make format && make check`, please.