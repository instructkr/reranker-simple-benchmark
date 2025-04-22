from .base import BaseReranker
from .huggingface_reranker import HuggingFaceReranker
from .flag_reranker import FlagRerankerType
from .flag_llm_reranker import FlagLLMRerankerType
from .flag_llm_lightweight_reranker import LayerWiseFlagLLMRerankerType
from .mxbai_reranker import MxbaiReranker


def determine_reranker_class(
    model_class: str, model_name: str, use_fp16: bool
) -> BaseReranker:
    """
    Decide which reranker class to instantiate based on model_class string.

    :param model_class: e.g. "huggingface", "flagreranker", "flagllmreranker", "flaglayerwise", "mxbai".
    :param model_name: Model path or name, e.g. "BAAI/bge-reranker-v2-gemma" or "mixedbread-ai/mxbai-rerank-base-v2".
    :param use_fp16: Whether to enable FP16.
    :return: An instance of the chosen reranker class.
    """
    model_class_lower: str = model_class.lower()

    if model_class_lower == "huggingface":
        reranker = HuggingFaceReranker(model_path=model_name, use_fp16=use_fp16)
    elif model_class_lower == "flagreranker":
        reranker = FlagRerankerType(model_path=model_name, use_fp16=use_fp16)
    elif model_class_lower == "flagllmreranker":
        reranker = FlagLLMRerankerType(model_path=model_name, use_fp16=use_fp16)
    elif model_class_lower == "flaglayerwise":
        reranker = LayerWiseFlagLLMRerankerType(
            model_path=model_name, use_fp16=use_fp16
        )
    elif model_class_lower == "mxbai":
        reranker = MxbaiReranker(model_path=model_name, use_fp16=use_fp16)
    else:
        print(
            f"Unknown model class; falling back to HuggingFaceReranker: {model_class}"
        )
        reranker = HuggingFaceReranker(model_path=model_name, use_fp16=use_fp16)

    return reranker
