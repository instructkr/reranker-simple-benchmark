from abc import ABC, abstractmethod
from typing import List, Tuple

__all__ = [
    "BaseRerankerWrapper",
    "BGEGemmaRerankerWrapper",
    "MxbaiRerankerWrapper",
    "Qwen3RerankerWrapper",
]


class BaseRerankerWrapper(ABC):
    """
    Abstract base class for reranker wrappers used in MTEB evaluation.
    All wrappers should implement the predict method to be compatible with MTEB.
    """

    @abstractmethod
    def predict(self, sentences: List[Tuple[str, str]], **kwargs) -> List[float]:
        """
        Compute relevance scores for query-document pairs.

        Args:
            sentences: List of (query, document) tuples
            **kwargs: Additional arguments like batch_size, show_progress_bar, etc.

        Returns:
            List of float scores, one per pair
        """
        pass


class Qwen3RerankerWrapper(BaseRerankerWrapper):
    def __init__(self, model_name: str, **kwargs):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, **kwargs)
        self.qwen3_instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    def _format_query(self, query: str) -> str:
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        return f"{prefix}<Instruct>: {self.qwen3_instruction}\n<Query>: {query}\n"

    def _format_document(self, document: str) -> str:
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return f"<Document>: {document}{suffix}"

    def predict(self, sentences: List[Tuple[str, str]], **kwargs) -> List[float]:
        formatted_sentences = []
        for query, document in sentences:
            if document is None:
                document = ""
            formatted_query = self._format_query(query)
            formatted_document = self._format_document(document)
            formatted_sentences.append([formatted_query, formatted_document])

        return self.model.predict(formatted_sentences, **kwargs)


class MxbaiRerankerWrapper(BaseRerankerWrapper):
    def __init__(self, model_name: str, **kwargs):
        from mxbai_rerank import MxbaiRerankV2

        self.model = MxbaiRerankV2(model_name, **kwargs)

    def predict(self, sentences: List[Tuple[str, str]], **kwargs) -> List[float]:
        queries, documents, _ = zip(*sentences)
        scores = self.model.predict(queries, documents, **kwargs)
        return scores


class BGEGemmaRerankerWrapper(BaseRerankerWrapper):
    def __init__(self, model_name: str, **kwargs):
        from FlagEmbedding import FlagLLMReranker

        self.model = FlagLLMReranker(model_name, **kwargs)

    def predict(self, sentences: List[Tuple[str, str]], **kwargs) -> List[float]:
        scores = self.model.compute_score(sentences, **kwargs)
        return scores
