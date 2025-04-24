import numpy as np
from FlagEmbedding import FlagReranker
from .base import BaseReranker


class FlagRerankerType(BaseReranker):
    """
    A multi-GPU capable Reranker based on the FlagEmbedding library.
    Uses FlagAutoReranker.from_finetuned(...) under the hood.
    """

    def __init__(
        self,
        model_path: str = "BAAI/bge-reranker-large",
        use_fp16: bool = True,
        cache_dir: str | None = None,
        apply_minmax_normalize: bool = False,
    ):
        """
        :param model_path: Path or name of the model on HF Hub (e.g. 'BAAI/bge-reranker-large').
        :param use_fp16: Whether to load/run the model with FP16.
        :param batch_size: Batch size for inference.
        :param query_max_length: Maximum length for tokenizing queries.
        :param doc_max_length: Maximum length for tokenizing docs (FlagEmbedding uses 'max_length' param).
        :param devices: list of device strings (e.g. ["cuda:0", "cuda:1"]). If None, defaults to single GPU or CPU.
        :param cache_dir: Optional cache directory (e.g. os.getenv('HF_HUB_CACHE')).
        :param apply_minmax_normalize: Whether to apply min-max normalization on the final scores.
        """

        # Internally, FlagAutoReranker handles multi-GPU, batch processing, etc.
        self.reranker = FlagReranker(
            model_path,
            use_fp16=use_fp16,
            cache_dir=cache_dir,
        )

        # If you want additional normalization after each "compute_score" call
        self.apply_minmax_normalize = apply_minmax_normalize

    def compute_score(
        self, pairs: list[tuple[str, str]], normalize: bool = True
    ) -> list[float]:
        """
        Compute score for each (query, doc_text) pair using FlagAutoReranker.

        The library expects pairs in the format: [ [query, doc], [query, doc], ... ]
        It handles multi-GPU if 'devices' was set with multiple GPUs.

        :param pairs: list of (query, doc_text).
        :param normalize: Whether to apply min-max normalization after getting raw scores.
        :return: list of float scores, one per (query, doc) pair.
        """
        input_pairs = [list(p) for p in pairs]  # convert tuple->list if needed
        scores = self.reranker.compute_score(input_pairs)

        if normalize or self.apply_minmax_normalize:
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))
            if (max_score - min_score) > 1e-8:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                scores = [0.0 for _ in scores]

        return scores

    def compute_score_batch(
        self, query: str, docs: list[str], normalize: bool = False
    ) -> list[float]:
        """
        Batch version, accepting a single query but multiple docs.
        This is needed by 'batched_compute_score' in your evaluate_model code.

        :param query: A single query string.
        :param docs: list of doc_texts.
        :param normalize: Whether to apply min-max normalization across these scores.
        :return: list of float scores, one per doc in 'docs'.
        """
        # Construct pairs [ [query, doc1], [query, doc2], ... ]
        input_pairs = [[query, doc] for doc in docs]
        scores = self.reranker.compute_score(input_pairs)

        if normalize or self.apply_minmax_normalize:
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))
            if (max_score - min_score) > 1e-8:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                scores = [0.0 for _ in scores]

        return scores
