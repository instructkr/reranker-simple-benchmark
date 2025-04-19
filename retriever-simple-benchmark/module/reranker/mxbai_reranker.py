import numpy as np
from mxbai_rerank import MxbaiRerankV2
from .base import BaseReranker


class MxbaiReranker(BaseReranker):
    def __init__(self, model_path, use_fp16=True, max_length=512):
        """
        Initialize MxbaiReranker with the given model path.

        :param model_path: Path or name of the model to use (e.g., "mixedbread-ai/mxbai-rerank-base-v2")
        :param use_fp16: Whether to use FP16 (not used by MxbaiRerankV2 but kept for API consistency)
        :param max_length: Maximum sequence length (not used by MxbaiRerankV2 but kept for API consistency)
        """
        # Import warnings here to suppress warnings from MxbaiRerankV2
        import warnings

        warnings.filterwarnings("ignore")

        self.model = MxbaiRerankV2(model_path)
        # Note: use_fp16 and max_length are ignored as MxbaiRerankV2 doesn't expose these options directly

    def compute_score_batch(
        self, query: str, doc_texts: list[str], normalize=False
    ) -> list[float]:
        """
        Compute scores for a batch of docs at once.

        :param query: The query string
        :param doc_texts: List of document texts
        :param normalize: Whether to normalize scores
        :return: List of scores for each document
        """
        # Use the MxbaiRerankV2's rank method which handles batching internally
        results = self.model.rank(query, doc_texts)

        # Extract scores from the results and map to the original document order
        # MxbaiRerankV2 returns results sorted by score, we need to remap to original order
        scores = [0.0] * len(doc_texts)
        for result in results:
            scores[result.index] = result.score

        # Normalize scores if requested
        if normalize and scores:
            min_s = float(np.min(scores))
            max_s = float(np.max(scores))
            if (max_s - min_s) > 1e-8:
                scores = [(s - min_s) / (max_s - min_s) for s in scores]
            else:
                scores = [0.0 for _ in scores]

        return scores

    def compute_score(self, pairs, normalize=True) -> list[float]:
        """
        Compute scores for a list of (query, document) pairs.

        :param pairs: List of (query, document) tuples
        :param normalize: Whether to normalize the scores
        :return: List of scores, one per pair
        """
        scores = []
        for q, doc in pairs:
            # When there's only one document, rank will assign index 0
            # So we can just take the score directly
            results = self.model.rank(q, [doc])
            scores.append(results[0].score)

        if normalize and scores:
            min_s = float(np.min(scores))
            max_s = float(np.max(scores))
            if (max_s - min_s) > 1e-8:
                scores = [(s - min_s) / (max_s - min_s) for s in scores]
            else:
                scores = [0.0 for _ in scores]

        return scores
