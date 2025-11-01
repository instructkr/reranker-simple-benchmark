from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """
    Abstract base class defining the interface for any Reranker.
    """

    @abstractmethod
    def compute_score(
        self, pairs: list[tuple[str, str]], normalize: bool = True
    ) -> list[float]:
        """
        Compute a single score for each (query, doc) pair.

        :param pairs: list of (query, doc_text).
        :param normalize: whether to apply any normalization after scoring.
        :return: list of float scores, one per pair
        """
        pass

    @abstractmethod
    def compute_score_batch(
        self, query: str, docs: list[str], normalize: bool = False
    ) -> list[float]:
        """
        Compute multiple scores for a single query + multiple docs.

        :param query: single query string
        :param docs: list of doc_texts
        :param normalize: whether to apply any normalization across these scores
        :return: list of float scores, one per doc
        """
        pass
