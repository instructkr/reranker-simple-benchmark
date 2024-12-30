# Retriever Simple Benchmark

## Dataset (Korean only)
* [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark/pull/6)
* (planned, not yet) [KURE](https://github.com/nlpai-lab/KURE)

## Models
* HuggingFace & FlagReranker supported cross-encoder
* (planned, not yet) HuggingFace & FlagEmbedding supported bi-encoder

## Command
### Setup
```
make init
```

### Run
```
make run TYPE=cross-encoder MODEL_NAME=sigridjineth/ko-reranker-v1.1 MODEL_CLASS=huggingface DATATYPE_NAME=AutoRAG
```

### Run with `torchrun`
```
make torchrun NPROC=2 TYPE=cross-encoder MODEL_NAME=dragonkue/bge-reranker-v2-m3-ko MODEL_CLASS=flagreranker DATATYPE_NAME=AutoRAG
```

## Reference

https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko/blob/main/cross_encoder_eval.ipynb