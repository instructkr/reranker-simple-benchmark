# Retriever Simple Benchmark
## Purpose
* The goal is to create something lightweight, with minimal dependencies, that runs effortlessly and delivers immediate results.

## Benchmark Results
* See [README.md](https://github.com/instructkr/retriever-simple-benchmark/blob/main/results/README.md)

## Dataset
* The target language is Korean at this moment.
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
# single GPU only at the moment.
make run TYPE=cross-encoder MODEL_NAME=sigridjineth/ko-reranker-v1.1 MODEL_CLASS=huggingface DATATYPE_NAME=AutoRAG
```

## Reference

https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko/blob/main/cross_encoder_eval.ipynb