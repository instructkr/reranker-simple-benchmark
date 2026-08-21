#!/usr/bin/env python3
"""
Stage 1 BM25 Retrieval Script for Korean Reranker Benchmark

This script performs BM25-based first-stage retrieval for all 10 benchmark datasets:
- 8 MTEB datasets: Ko-StrategyQA, AutoRAGRetrieval, MIRACLRetrieval, PublicHealthQA,
                   BelebeleRetrieval, MrTidyRetrieval, MultiLongDocRetrieval, XPQARetrieval
- 2 Custom datasets: SQuADKorV1Retrieval, WebFAQRetrieval

Supports multiple Korean tokenizers: Mecab (recommended), Kiwi, Okt, Kkma

Usage:
    # All 10 datasets with Mecab tokenizer
    uv run python eval/retrieve_stage1_bm25.py --tokenizer Mecab --datasets all

    # Specific datasets with Kiwi tokenizer
    uv run python eval/retrieve_stage1_bm25.py --tokenizer Kiwi --datasets Ko-StrategyQA AutoRAGRetrieval

    # Custom datasets only
    uv run python eval/retrieve_stage1_bm25.py --tokenizer Mecab --datasets SQuADKorV1Retrieval WebFAQRetrieval
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s
from bm25s.tokenization import Tokenizer, Tokenized
from datasets import load_dataset
from tqdm import tqdm

import mteb
# Method 1(mteb 프레임워크 기반 BM25)은 mteb 1.38 API 를 사용한다. mteb 2.x 에는 아래 심볼이 없으므로
# guard 하여 모듈 import 가 되게 한다(Method 1 은 그 env 에서 비활성; Method 2 / mteb-2.x qrels 마이닝은 정상).
try:
    from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
    from mteb.model_meta import ModelMeta
    from mteb.models.wrapper import Wrapper
    from mteb.requires_package import requires_package
    _MTEB_V1_API = True
except Exception:
    _MTEB_V1_API = False

# Korean tokenizers
from konlpy.tag import Kkma, Okt
from kiwipiepy import Kiwi

try:
    from konlpy.tag import Mecab
    MECAB_AVAILABLE = True
except Exception:
    MECAB_AVAILABLE = False
    print("Warning: konlpy MeCab is not available")

try:
    import mecab as _pymecab  # python-mecab-ko (mecab-ko-dic 번들 = konlpy Mecab 과 동일 사전)
    PYMECAB_AVAILABLE = True
except Exception:
    PYMECAB_AVAILABLE = False


class _PyMecabWrapper:
    """python-mecab-ko 를 konlpy Mecab 과 동일한 .morphs 인터페이스로 감싼다."""

    def __init__(self):
        self._m = _pymecab.MeCab()

    def morphs(self, text: str):
        return self._m.morphs(text)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dataset configuration
MTEB_DATASETS = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "MIRACLRetrieval",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "XPQARetrieval",
]

CUSTOM_DATASETS = [
    "SQuADKorV1Retrieval",
    "WebFAQRetrieval",
]

ALL_DATASETS = MTEB_DATASETS + CUSTOM_DATASETS


def clean_text(text: str) -> str:
    """Clean text by replacing unsupported characters"""
    pattern = r"[^가-힣a-zA-Z0-9\s~!@#$%^&*()_\-+=\[\]{}|\\;:'\",.<>/?`]"
    cleaned = re.sub(pattern, "-", text)
    return cleaned


class Kiwi_(Kiwi):
    """Kiwi tokenizer wrapper with morphs method"""
    def morphs(self, text: str) -> List[str]:
        return [morph.form for morph in self.tokenize(text)]


class KoreanTokenizer(Tokenizer):
    """Korean tokenizer wrapper for BM25s compatibility"""

    def __init__(self, tokenizer_name: str = "Mecab", stopwords=None):
        super().__init__(stopwords=stopwords)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = self._init_tokenizer(tokenizer_name)
        logger.info(f"Initialized {tokenizer_name} tokenizer")

    def _init_tokenizer(self, tokenizer_name: str):
        """Initialize the specified Korean tokenizer"""
        if tokenizer_name == "Mecab":
            # konlpy Mecab(시스템 mecab-ko-dic) 우선, 실패 시 python-mecab-ko(mecab-ko-dic 번들)로 대체.
            # 두 경로 모두 mecab-ko-dic 를 사용하므로 토큰화가 동일하다(기존 pool 과 일관).
            if MECAB_AVAILABLE:
                try:
                    return Mecab()
                except Exception as e:
                    logger.warning(f"konlpy Mecab init 실패({e}) → python-mecab-ko 로 대체")
            if PYMECAB_AVAILABLE:
                return _PyMecabWrapper()
            raise ImportError("MeCab 사용 불가: konlpy Mecab / python-mecab-ko 모두 없음")
        elif tokenizer_name == "Kiwi":
            return Kiwi_()
        elif tokenizer_name == "Okt":
            return Okt()
        elif tokenizer_name == "Kkma":
            return Kkma()
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    def _get_morphs(self, text: str) -> List[str]:
        """Get morphemes from text using the initialized tokenizer"""
        try:
            return self.tokenizer.morphs(text)
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error, cleaning text: {e}")
            return self.tokenizer.morphs(clean_text(text))

    def tokenize(self, texts: List[str], return_ids: bool = True) -> Tokenized | List[List[str]]:
        """
        Tokenize texts into morphemes

        Args:
            texts: List of texts to tokenize
            return_ids: If True, return Tokenized object with IDs; if False, return token strings

        Returns:
            Tokenized object or list of token lists
        """
        # Tokenize each text
        corpus_tokens = []
        for text in texts:
            try:
                tokens = self._get_morphs(text)
                corpus_tokens.append(tokens)
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
                corpus_tokens.append([])

        if not return_ids:
            return corpus_tokens

        # Build vocabulary
        vocab = {"": 0}
        index = 1
        for tokens in corpus_tokens:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = index
                    index += 1

        # Convert tokens to indices
        indexed_tokens = [
            [vocab[token] for token in tokens] for tokens in corpus_tokens
        ]

        return Tokenized(ids=indexed_tokens, vocab=vocab)


# ============================================================================
# Method 1: MTEB-based retrieval (for 8 MTEB datasets)
# ============================================================================

def bm25_loader(**kwargs):
    """BM25 loader for MTEB evaluation framework"""
    model_name = kwargs.get("model_name", "BM25")
    task_name = kwargs.get("task_name", "bm25s")
    tokenizer_name = kwargs.get("tokenizer_name", "Mecab")
    top_k = kwargs.get("top_k", 1000)

    requires_package(bm25_loader, "bm25s", model_name, "pip install mteb[bm25s]")
    import bm25s
    import Stemmer

    class BM25Search(DRESModel, Wrapper):
        """BM25 search model for MTEB"""

        def __init__(
            self,
            previous_results: str = None,
            stopwords: str = "en",
            stemmer_language: str | None = "english",
            **kwargs,
        ):
            super().__init__(
                model=None,
                batch_size=1,
                corpus_chunk_size=1,
                previous_results=previous_results,
                **kwargs,
            )

            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )
            self.task_name = task_name
            self.tokenizer_name = tokenizer_name
            self.top_k = top_k

        @classmethod
        def name(cls):
            return "bm25s"

        def search(
            self,
            corpus: dict[str, dict[str, str]],
            queries: dict[str, str | list[str]],
            top_k: int,
            score_function: str,
            return_sorted: bool = False,
            **kwargs,
        ) -> dict[str, dict[str, float]]:
            logger.info("Encoding Corpus...")
            corpus_ids = list(corpus.keys())
            corpus_with_ids = [
                {
                    "doc_id": cid,
                    **(
                        {"text": corpus[cid]}
                        if isinstance(corpus[cid], str)
                        else corpus[cid]
                    ),
                }
                for cid in corpus_ids
            ]

            corpus_texts = [
                "\n".join([doc.get("title", ""), doc["text"]])
                for doc in corpus_with_ids
            ]
            encoded_corpus = self.encode(corpus_texts, task_name=self.task_name)

            logger.info(
                f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, "
                f"{len(encoded_corpus.vocab):,} vocab"
            )

            # Create BM25 model and index corpus
            retriever = bm25s.BM25()
            retriever.index(encoded_corpus)

            logger.info("Encoding Queries...")
            query_ids = list(queries.keys())
            self.results = {qid: {} for qid in query_ids}
            queries_texts = [queries[qid] for qid in queries]

            query_token_strs = self.encode(queries_texts, return_ids=False)

            logger.info(f"Retrieving Results... {len(queries):,} queries")

            queries_results, queries_scores = retriever.retrieve(
                query_token_strs, corpus=corpus_with_ids, k=self.top_k
            )

            # Process results
            for qi, qid in enumerate(query_ids):
                query_results = queries_results[qi]
                scores = queries_scores[qi]
                doc_id_to_score = {}

                for ri in range(len(query_results)):
                    doc = query_results[ri]
                    score = scores[ri]
                    doc_id = doc["doc_id"]
                    doc_id_to_score[doc_id] = float(score)

                self.results[qid] = doc_id_to_score

            return self.results

        def encode(self, texts: list[str], task_name=None, return_ids: bool = True, **kwargs):
            """Encode input text as term vectors"""
            tokenizer = KoreanTokenizer(self.tokenizer_name, stopwords=None)
            return tokenizer.tokenize(texts, return_ids=return_ids)

    return BM25Search(**kwargs)


def retrieve_mteb_dataset(
    dataset_name: str,
    tokenizer_name: str = "Mecab",
    output_folder: str = "eval/results/stage1/bm25_Mecab",
    top_k: int = 1000,
):
    """
    Retrieve using MTEB framework for standard MTEB datasets

    Args:
        dataset_name: MTEB dataset name
        tokenizer_name: Korean tokenizer to use
        output_folder: Output directory for results
        top_k: Number of documents to retrieve
    """
    logger.info(f"{'='*80}")
    logger.info(f"Processing MTEB dataset: {dataset_name}")
    logger.info(f"{'='*80}")

    # Get MTEB tasks
    tasks = mteb.get_tasks(tasks=[dataset_name], languages=["kor"])

    # Check corpus size to adjust top_k
    if hasattr(tasks[0], "load_data"):
        tasks[0].load_data()

    len_data = None
    corpus = tasks[0].corpus

    # Try to get corpus size
    if isinstance(corpus, dict):
        for key, value in corpus.items():
            if isinstance(value, dict):
                for split_key in ["test", "dev", "train"]:
                    if split_key in value and value[split_key] is not None:
                        try:
                            len_data = len(value[split_key])
                            break
                        except:
                            pass
                if len_data:
                    break
            elif value is not None:
                try:
                    len_data = len(value)
                    break
                except:
                    pass

    # Special handling for known small corpora
    if dataset_name == "XPQARetrieval":
        len_data = 889

    if len_data is None or len_data >= 1000:
        if len_data is None:
            len_data = 1000
        logger.info(f"Corpus size: {len_data:,} (using top_k={top_k})")
    else:
        logger.info(f"Corpus size: {len_data:,} (adjusting top_k={len_data})")

    # Create BM25 model
    bm25_model = ModelMeta(
        loader=partial(
            bm25_loader,
            model_name="bm25s",
            task_name="bm25s",
            tokenizer_name=tokenizer_name,
            top_k=len_data if len_data < 1000 else top_k,
        ),
        name="bm25s",
        languages=["kor-Hang"],
        open_weights=True,
        revision="0_1_10",
        release_date="2024-07-10",
        n_parameters=None,
        memory_usage_mb=None,
        embed_dim=None,
        license=None,
        max_tokens=None,
        reference="https://github.com/xhluca/bm25s",
        similarity_fn_name=None,
        framework=[],
        use_instructions=False,
        public_training_code="https://github.com/xhluca/bm25s",
        public_training_data=None,
        training_datasets=None,
    ).load_model()

    # Run evaluation
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        bm25_model,
        output_folder=output_folder,
        save_predictions=True,
    )

    logger.info(f"Completed: {dataset_name}")
    logger.info(f"Scores: {results[0].scores}")
    logger.info("")


# ============================================================================
# Method 2: Direct retrieval with qrels incorporation (for custom datasets)
# ============================================================================

def load_custom_dataset(dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Load custom dataset from HuggingFace

    Args:
        dataset_name: "SQuADKorV1Retrieval" or "WebFAQRetrieval"

    Returns:
        queries: {query_id: query_text}
        corpus: {doc_id: doc_text}
        qrels: {query_id: [relevant_doc_ids]}
    """
    logger.info(f"Loading custom dataset: {dataset_name}...")

    if dataset_name == "SQuADKorV1Retrieval":
        hf_dataset = "yjoonjang/squad_kor_v1"
    elif dataset_name == "WebFAQRetrieval":
        hf_dataset = "PaDaS-Lab/webfaq-retrieval"
    else:
        raise ValueError(f"Unknown custom dataset: {dataset_name}")

    # Load queries
    if dataset_name == "WebFAQRetrieval":
        queries_ds = load_dataset(hf_dataset, 'kor-queries', split='test')
    else:
        queries_ds = load_dataset(hf_dataset, 'queries', split='test')
    queries = {item['_id']: item['text'] for item in queries_ds}
    logger.info(f"Loaded {len(queries):,} queries")

    # Load corpus
    if dataset_name == "WebFAQRetrieval":
        corpus_ds = load_dataset(hf_dataset, 'kor-corpus', split='corpus')
    else:
        corpus_ds = load_dataset(hf_dataset, 'corpus', split='test')

    corpus = {}
    for item in corpus_ds:
        text = item['text']
        if item.get('title'):
            text = f"{item['title']}\n{text}"
        corpus[item['_id']] = text
    logger.info(f"Loaded {len(corpus):,} documents")

    # Load qrels
    if dataset_name == "WebFAQRetrieval":
        qrels_ds = load_dataset(hf_dataset, 'kor-qrels', split='test')
    else:
        qrels_ds = load_dataset(hf_dataset, 'default', split='test')

    qrels = {}
    for item in qrels_ds:
        query_id = item['query-id']
        corpus_id = item['corpus-id']
        if query_id not in qrels:
            qrels[query_id] = []
        qrels[query_id].append(corpus_id)
    logger.info(f"Loaded qrels for {len(qrels):,} queries")

    return queries, corpus, qrels


def retrieve_custom_dataset(
    dataset_name: str,
    tokenizer_name: str = "Mecab",
    output_folder: str = "eval/results/stage1/top_1k_qrels",
    top_k: int = 1000,
):
    """
    Retrieve using direct BM25 for custom datasets with qrels integration

    This method ensures ground truth documents are included in the top-k results
    by placing them at the front of the results list.

    Args:
        dataset_name: Custom dataset name
        tokenizer_name: Korean tokenizer to use
        output_folder: Output directory for results
        top_k: Number of documents to retrieve
    """
    logger.info(f"{'='*80}")
    logger.info(f"Processing custom dataset: {dataset_name}")
    logger.info(f"{'='*80}")
    queries, corpus, qrels = load_custom_dataset(dataset_name)
    _bm25_retrieve_and_save(dataset_name, queries, corpus, qrels, tokenizer_name, output_folder, top_k)


def _bm25_retrieve_and_save(dataset_name, queries, corpus, qrels, tokenizer_name, output_folder, top_k):
    """공용 BM25 검색 코어: (queries, corpus, qrels) → 정답 prepend + BM25 top-k → <dataset>_id.jsonl.
    custom 데이터셋(load_custom_dataset)과 mteb 2.x task(load_mteb_kor_task) 양쪽에서 재사용한다."""
    tokenizer = KoreanTokenizer(tokenizer_name, stopwords=None)

    logger.info("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    encoded_corpus = tokenizer.tokenize(corpus_texts, return_ids=True)
    logger.info(f"Encoded {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab")

    retriever = bm25s.BM25()
    retriever.index(encoded_corpus)

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_tokens = tokenizer.tokenize(query_texts, return_ids=False)

    actual_top_k = min(top_k, len(corpus))
    logger.info(f"Retrieving top {actual_top_k} for {len(queries):,} queries...")
    bm25_results, _ = retriever.retrieve(query_tokens, corpus=corpus_ids, k=actual_top_k)

    results = {}
    for qi, qid in enumerate(tqdm(query_ids, desc="Processing queries")):
        relevant_docs = list(qrels.get(qid, []))
        rel_set = set(relevant_docs)
        final_docs = list(relevant_docs)  # 정답을 앞에 배치(gold prepend)
        for doc_id in bm25_results[qi]:
            if doc_id not in rel_set:
                final_docs.append(doc_id)
                if len(final_docs) >= actual_top_k:
                    break
        results[qid] = final_docs[:actual_top_k]

    output_path = Path(output_folder) / f"{dataset_name}_id.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for query_id, doc_ids in results.items():
            f.write(json.dumps({"query_id": query_id, "relevance_ids": doc_ids}, ensure_ascii=False) + "\n")
    logger.info(f"Completed: {dataset_name} → {output_path} ({len(results)} queries)")
    logger.info("")


def _doc_text_from(v) -> str:
    if isinstance(v, dict):
        title = v.get("title") or ""
        text = v.get("text") or ""
        return (title + "\n" + text).strip() if title else text
    return v if isinstance(v, str) else str(v)


def load_mteb_kor_task(task_name: str):
    """mteb 2.x 로 한국어 retrieval task 를 로드해 (queries, corpus, qrels[list]) 딕셔너리로 반환.
    task 의 모든 (kor subset, split) 을 병합한다(예: MultiLongDocRetrieval = dev+test).
    공식 MTEB(kor, v2) split 기준을 따른다(각 task 표준 eval_splits)."""
    task = None
    for kw in (dict(tasks=[task_name], languages=["kor-Hang"]), dict(tasks=[task_name])):
        try:
            ts = mteb.get_tasks(**kw)
            if ts:
                task = ts[0]
                break
        except Exception:
            pass
    if task is None:
        raise ValueError(f"mteb task 없음: {task_name}")
    task.load_data()
    task.convert_v1_dataset_format_to_v2(num_proc=1)
    ds = task.dataset
    queries, corpus, qrels = {}, {}, {}
    for sub in ds:
        for sp in ds[sub]:
            blk = ds[sub][sp]
            c, q, r = blk["corpus"], blk["queries"], blk["relevant_docs"]
            if hasattr(c, "keys"):
                for k, v in c.items():
                    corpus[str(k)] = _doc_text_from(v)
            else:
                for row in c:
                    corpus[str(row["id"])] = _doc_text_from(row)
            if hasattr(q, "keys"):
                for k, v in q.items():
                    queries[str(k)] = v if isinstance(v, str) else v.get("text", "")
            else:
                for row in q:
                    queries[str(row["id"])] = row.get("text", "")
            for qid, docs in r.items():
                qrels[str(qid)] = [str(d) for d in docs]
            logger.info(f"  [{sub}/{sp}] q={len(q)} corpus={len(c)}")
    return queries, corpus, qrels


def retrieve_kor_task(
    task_name: str,
    tokenizer_name: str = "Mecab",
    output_folder: str = "eval/results/stage1/top_1k_qrels",
    top_k: int = 1000,
):
    """mteb 2.x 기반: 공식 kor 벤치 task 하나를 BM25 마이닝해 <task>_id.jsonl 생성.
    MultiLongDocRetrieval(dev+test), LawIRKo 등 신규/누락 pool 생성에 사용."""
    logger.info(f"{'='*80}")
    logger.info(f"Processing mteb(kor) task: {task_name}")
    logger.info(f"{'='*80}")
    queries, corpus, qrels = load_mteb_kor_task(task_name)
    _bm25_retrieve_and_save(task_name, queries, corpus, qrels, tokenizer_name, output_folder, top_k)


# ============================================================================
# Main execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 BM25 Retrieval for Korean Reranker Benchmark (10 datasets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All 10 datasets with Mecab (recommended)
  python eval/retrieve_stage1_bm25.py --tokenizer Mecab --datasets all

  # Specific MTEB datasets
  python eval/retrieve_stage1_bm25.py --tokenizer Mecab --datasets Ko-StrategyQA AutoRAGRetrieval

  # Custom datasets only
  python eval/retrieve_stage1_bm25.py --tokenizer Mecab --datasets SQuADKorV1Retrieval WebFAQRetrieval

  # Compare tokenizers
  python eval/retrieve_stage1_bm25.py --tokenizer Kiwi --datasets all --output_dir eval/results/stage1/bm25_Kiwi
        """
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Mecab",
        choices=["Mecab", "Kiwi", "Okt", "Kkma"],
        help="Korean tokenizer to use (default: Mecab, recommended based on benchmark results)"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Datasets to process. Options: 'all', 'mteb', 'custom', or specific dataset names. "
            f"Available: {', '.join(ALL_DATASETS)}"
        )
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of documents to retrieve (default: 1000)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for results. "
            "If not specified, uses 'eval/results/stage1/bm25_{tokenizer}' for MTEB datasets "
            "and 'eval/results/stage1/top_1k_qrels' for custom datasets"
        )
    )

    parser.add_argument(
        "--kor_tasks",
        nargs="+",
        default=None,
        help=(
            "mteb 2.x 기반으로 지정한 한국어 retrieval task 를 BM25 마이닝해 top_1k_qrels/<task>_id.jsonl 생성. "
            "공식 MTEB(kor, v2) split 기준(각 task 표준 eval_splits, 예: MultiLongDocRetrieval=dev+test). "
            "예: --kor_tasks MultiLongDocRetrieval LawIRKo"
        ),
    )

    args = parser.parse_args()

    # mteb 2.x 기반 kor task 마이닝(신규/누락 pool): --kor_tasks 지정 시 이 경로만 실행
    if args.kor_tasks:
        output_folder = args.output_dir or "eval/results/stage1/top_1k_qrels"
        logger.info(f"[kor_tasks] mteb 2.x BM25 마이닝: {args.kor_tasks} (tokenizer={args.tokenizer}) → {output_folder}")
        for task_name in args.kor_tasks:
            try:
                retrieve_kor_task(
                    task_name=task_name,
                    tokenizer_name=args.tokenizer,
                    output_folder=output_folder,
                    top_k=args.top_k,
                )
            except Exception as e:
                logger.error(f"Failed to mine {task_name}: {e}")
                import traceback
                traceback.print_exc()
        logger.info("All kor_tasks processed.")
        return

    # Determine which datasets to process
    if "all" in args.datasets:
        datasets_to_process = ALL_DATASETS
    elif "mteb" in args.datasets:
        datasets_to_process = MTEB_DATASETS
    elif "custom" in args.datasets:
        datasets_to_process = CUSTOM_DATASETS
    else:
        datasets_to_process = args.datasets

    # Validate datasets
    invalid_datasets = [d for d in datasets_to_process if d not in ALL_DATASETS]
    if invalid_datasets:
        logger.error(f"Invalid datasets: {invalid_datasets}")
        logger.error(f"Available datasets: {ALL_DATASETS}")
        return

    logger.info(f"{'='*80}")
    logger.info(f"Stage 1 BM25 Retrieval")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Datasets: {datasets_to_process}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"{'='*80}\n")

    # Process each dataset
    for dataset_name in datasets_to_process:
        try:
            if dataset_name in CUSTOM_DATASETS:
                # Custom datasets: use direct retrieval with qrels
                output_folder = args.output_dir or "eval/results/stage1/top_1k_qrels"
                retrieve_custom_dataset(
                    dataset_name=dataset_name,
                    tokenizer_name=args.tokenizer,
                    output_folder=output_folder,
                    top_k=args.top_k,
                )
            else:
                # MTEB datasets: use MTEB framework
                output_folder = args.output_dir or f"eval/results/stage1/bm25_{args.tokenizer}"
                retrieve_mteb_dataset(
                    dataset_name=dataset_name,
                    tokenizer_name=args.tokenizer,
                    output_folder=output_folder,
                    top_k=args.top_k,
                )
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"{'='*80}")
    logger.info("All datasets processed successfully!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
