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
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

# Korean tokenizers
from konlpy.tag import Kkma, Okt
from kiwipiepy import Kiwi

try:
    from konlpy.tag import Mecab
    MECAB_AVAILABLE = True
except Exception:
    MECAB_AVAILABLE = False
    print("Warning: MeCab is not available")

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
            if not MECAB_AVAILABLE:
                raise ImportError("MeCab is not available")
            mecab_dict_path = "ko-reranker-benchmark/mecab/lib/mecab/dic/mecab-ko-dic"
            return Mecab(dicpath=mecab_dict_path)
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

    # Load dataset
    queries, corpus, qrels = load_custom_dataset(dataset_name)

    # Initialize tokenizer
    tokenizer = KoreanTokenizer(tokenizer_name, stopwords=None)

    # Encode corpus
    logger.info("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    encoded_corpus = tokenizer.tokenize(corpus_texts, return_ids=True)
    logger.info(f"Encoded {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab")

    # Build BM25 index
    logger.info("Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(encoded_corpus)

    # Encode queries
    logger.info("Encoding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_tokens = tokenizer.tokenize(query_texts, return_ids=False)

    # Adjust top_k if corpus is smaller
    actual_top_k = min(top_k, len(corpus))
    if actual_top_k < top_k:
        logger.warning(
            f"Corpus size ({len(corpus):,}) is smaller than top_k ({top_k}). "
            f"Using top_k={actual_top_k}"
        )

    # Perform retrieval
    logger.info(f"Retrieving top {actual_top_k} results for {len(queries):,} queries...")
    results = {}

    bm25_results, bm25_scores = retriever.retrieve(
        query_tokens,
        corpus=corpus_ids,
        k=actual_top_k
    )

    # Process results: place ground truth docs first
    for qi, qid in enumerate(tqdm(query_ids, desc="Processing queries")):
        # Get ground truth document IDs
        relevant_docs = qrels.get(qid, [])

        # Get BM25 results
        bm25_docs = [doc_id for doc_id in bm25_results[qi]]

        # Place ground truth documents first
        final_docs = list(relevant_docs)

        # Add BM25 results (excluding ground truth docs already added)
        for doc_id in bm25_docs:
            if doc_id not in relevant_docs:
                final_docs.append(doc_id)
                if len(final_docs) >= actual_top_k:
                    break

        # Ensure we have exactly top_k documents
        final_docs = final_docs[:actual_top_k]
        results[qid] = final_docs

    logger.info(f"Retrieved results for {len(results):,} queries")

    # Save results
    output_path = Path(output_folder) / f"{dataset_name}_id.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for query_id, doc_ids in results.items():
            line = json.dumps({
                "query_id": query_id,
                "relevance_ids": doc_ids
            }, ensure_ascii=False)
            f.write(line + '\n')

    logger.info(f"Completed: {dataset_name}")
    logger.info("")


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

    args = parser.parse_args()

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
