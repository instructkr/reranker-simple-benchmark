"""transformers 5.x 호환 shim.

transformers 5.x 는 `PreTrainedTokenizerBase.prepare_for_model` 을 제거했는데,
일부 리랭커 라이브러리(mxbai_rerank, FlagEmbedding LLM reranker)가 이를 내부에서 호출한다.
두 라이브러리 모두 동일한 단순 패턴으로만 사용한다:
    tokenizer.prepare_for_model(ids, pair_ids, truncation="only_second",
        max_length=M, padding=False, add_special_tokens=False,
        return_attention_mask=False, return_token_type_ids=False)
→ 이 패턴에 한해 원래 동작(특수토큰 없이 concat + only_second 절단)을 복원한다.
이렇게 하면 라이브러리의 *원본 스코어링 로직을 그대로* 쓸 수 있어 native 재구현으로 인한 편차가 없다.
"""
from __future__ import annotations


def _prepare_for_model(
    self,
    ids,
    pair_ids=None,
    *,
    add_special_tokens: bool = True,
    padding=False,
    truncation=None,
    max_length=None,
    stride: int = 0,
    return_tensors=None,
    return_token_type_ids=None,
    return_attention_mask=None,
    **kwargs,
):
    ids = list(ids)
    pair = list(pair_ids) if pair_ids is not None else []
    has_pair = pair_ids is not None

    # special tokens (mxbai/FlagEmbedding 는 add_special_tokens=False 로만 호출하므로 그 경로만 필요)
    if add_special_tokens:
        seq = self.build_inputs_with_special_tokens(ids, pair if has_pair else None)
        tt = self.create_token_type_ids_from_sequences(ids, pair if has_pair else None)
        # 특수토큰 경로에서는 절단을 지원하지 않음(호출자가 사용하지 않음)
        out = {"input_ids": seq}
        if return_token_type_ids:
            out["token_type_ids"] = tt
        if return_attention_mask:
            out["attention_mask"] = [1] * len(seq)
        return out

    # truncation (only_second / only_first / longest_first)
    if max_length is not None and truncation and truncation != "do_not_truncate":
        total = len(ids) + len(pair)
        overflow = total - max_length
        if overflow > 0:
            mode = "longest_first" if truncation is True else truncation
            if mode == "only_second":
                pair = pair[: max(0, len(pair) - overflow)]
            elif mode == "only_first":
                ids = ids[: max(0, len(ids) - overflow)]
            else:  # longest_first
                for _ in range(overflow):
                    if len(ids) > len(pair):
                        ids.pop()
                    else:
                        pair.pop()

    input_ids = ids + pair
    out = {"input_ids": input_ids}
    if return_token_type_ids:
        out["token_type_ids"] = [0] * len(ids) + [1] * len(pair)
    if return_attention_mask:
        out["attention_mask"] = [1] * len(input_ids)
    return out


def apply() -> bool:
    """prepare_for_model 이 없으면 shim 을 주입. 반환: 주입 여부."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "prepare_for_model"):
        PreTrainedTokenizerBase.prepare_for_model = _prepare_for_model
        return True
    return False
