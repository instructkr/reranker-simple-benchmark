"""
MTEB(2.x) 기반 Reranker 평가 — "옳은 평가"(정답 항상 후보 포함) 적용.

핵심(Approach 1, mteb-native):
  - mteb 2.x 의 native reranking(`SearchCrossEncoderWrapper` + `top_ranked`)을 그대로 사용해
    채점·집계(get_score, splits/subsets 평균)를 mteb 공식 경로로 유지한다.
  - 결함(1차검색 top-k 에 정답이 없으면 reranking 점수를 못 냄)만 교정: 각 query 재랭킹 후보를
    {corpus 내 정답 전부} ∪ {BM25 top-50 중 정답 제외} 로 구성해 정답을 항상 포함시킨다.
    → `AbsTaskRetrieval._evaluate_subset` 에서 `data_split["top_ranked"]` 를 주입(monkey-patch).
  - 1차 pool = 기존 stage1 BM25 결과(`eval/results/stage1/top_1k_qrels/<task>_id.jsonl`).

모델: 표준 sentence-transformers CrossEncoder 는 mteb 가 자동 래핑. 비-ST(Qwen 프롬프트/Jina/
Nemotron/mxbai/bge-gemma)는 `MtebRerankAdapter` 로 mteb CrossEncoderProtocol 에 맞춘다.

사용:
  uv run python eval/evaluate_reranker.py --model_names BAAI/bge-reranker-v2-m3 --tasks AutoRAGRetrieval
  uv run python eval/evaluate_reranker.py            # 전체 모델 × 전체 task
"""
import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# transformers 5.x 에서 제거된 tokenizer.prepare_for_model 복원 (mxbai_rerank / FlagEmbedding 용).
# 모델/토크나이저 로드 전에 반드시 적용해야 한다.
from tf5_compat import apply as _apply_tf5_compat  # noqa: E402
_apply_tf5_compat()

V2_ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = V2_ROOT / "eval/results/stage1/top_1k_qrels"
OUT_DIR = V2_ROOT / "eval/results/stage2"
NEG_TOP_K = 50  # 정답 외 negative 후보로 사용할 BM25 상위 개수
MAX_LENGTH = 8192  # 전 모델 공통 최대 시퀀스 길이(공정 비교). 결과 json 에 _max_length 로 기록.

# 한국어 벤치 10종 (기존 리더보드와 동일 집합)
DEFAULT_TASKS = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MIRACLRetrieval",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "SQuADKorV1Retrieval",
    "LawIRKo",
]

DEFAULT_MODEL_NAMES = [
    # target 기존 14 중 tf5 호환 11 (arch-broken 3 제외: sigrid ko-reranker-v1.1 / gte-multilingual /
    # jina-reranker-v2 — 커스텀 arch 가 transformers 5.x 에서 device-assert/import 실패)
    "BAAI/bge-reranker-v2-m3",
    "dragonkue/bge-reranker-v2-m3-ko",
    "jinaai/jina-reranker-v3",
    "telepix/PIXIE-Spell-Reranker-Preview-0.6B",
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    "tomaarsen/Qwen3-Reranker-4B-seq-cls",
    "tomaarsen/Qwen3-Reranker-8B-seq-cls",
    "Dongjin-kr/ko-reranker",
    "upskyy/ko-reranker-8k",
    "mixedbread-ai/mxbai-rerank-large-v2",
    "BAAI/bge-reranker-v2-gemma",
    # 신규 6 (mteb 2.x + tf5 단일 env 로 통합)
    "jinaai/jina-reranker-v3.5",
    "nlpai-lab/LAMAR-600m",
    "nvidia/llama-nemotron-rerank-1b-v2",
    "cross-encoder/ettin-reranker-1b-v1",
    "zeroentropy/zerank-2-reranker",
    "lightonai/LightOn-rerank-PW-4B",
]

# 대부분 task 는 languages=["kor-Hang"] 로 한국어 subset 선택. 일부(모노링구얼 list eval_langs)는
# 필터 없이 로드해야 하므로 fallback.
KOR_LANG = ["kor-Hang"]


# ============================================================================
# gold-injection monkey-patch (정답 항상 후보 포함)
# ============================================================================
import mteb  # noqa: E402
from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: E402
from mteb.models.model_meta import ModelMeta  # noqa: E402

_pool_cache: dict[str, dict[str, list[str]]] = {}


def _load_pool(task_name: str) -> dict[str, list[str]]:
    if task_name in _pool_cache:
        return _pool_cache[task_name]
    path = POOL_DIR / f"{task_name}_id.jsonl"
    pool: dict[str, list[str]] = {}
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            pool[str(d["query_id"])] = [str(x) for x in d["relevance_ids"]]
    else:
        print(f"[gold-inject] WARNING: pool file 없음: {path}")
    _pool_cache[task_name] = pool
    return pool


def _ids(container) -> set:
    if hasattr(container, "keys"):
        return {str(k) for k in container.keys()}
    return {str(r.get("id") if hasattr(r, "get") else r) for r in container}


_orig_evaluate_subset = AbsTaskRetrieval._evaluate_subset


def _gold_inject_evaluate_subset(self, model, data_split, *, hf_split, hf_subset, **kwargs):
    """data_split["top_ranked"] 를 {정답 전부}∪{BM25 top-NEG_TOP_K} 로 교정."""
    pool = _load_pool(self.metadata.name)
    corpus = data_split["corpus"]
    queries = data_split["queries"]
    rels = data_split["relevant_docs"]

    cids = _ids(corpus)
    if hasattr(queries, "keys"):
        qids = [str(q) for q in queries.keys()]
    else:
        qids = [str(r.get("id") if hasattr(r, "get") else r) for r in queries]

    tr: dict[str, list[str]] = {}
    n_noneg = 0
    for qid in qids:
        gr = rels.get(qid, {}) if hasattr(rels, "get") else {}
        golds = [g for g in gr if g in cids]
        gset = set(golds)
        bm = [d for d in pool.get(qid, [])[:NEG_TOP_K] if d not in gset and d in cids]
        tr[qid] = golds + bm
        if not bm:
            n_noneg += 1
    data_split["top_ranked"] = tr
    self._top_k = max((len(v) for v in tr.values()), default=1) + 1
    msg = f"[gold-inject] {self.metadata.name}[{hf_subset}/{hf_split}] q={len(tr)} top_k={self._top_k}"
    if n_noneg:
        msg += f" (no-BM25-neg q={n_noneg})"
    print(msg, flush=True)
    return _orig_evaluate_subset(self, model, data_split, hf_split=hf_split, hf_subset=hf_subset, **kwargs)


AbsTaskRetrieval._evaluate_subset = _gold_inject_evaluate_subset


# ============================================================================
# 비-ST 모델용 mteb CrossEncoderProtocol 어댑터
#   내부 wrapper 는 predict([(q,d), ...]) -> list[float] 인터페이스만 구현하면 된다.
# ============================================================================
class MtebRerankAdapter:
    def __init__(self, inner, name: str, batch_size: int = 16):
        self.inner = inner
        self.batch_size = batch_size
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=name, revision="gold-inject", loader=None)
        )

    @staticmethod
    def _texts(loader) -> list[str]:
        return [t for batch in loader for t in batch["text"]]

    def predict(self, inputs1, inputs2, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        queries = self._texts(inputs1)
        corpus = self._texts(inputs2)
        pairs = list(zip(queries, corpus))
        try:
            scores = self.inner.predict(pairs, batch_size=self.batch_size)
        except TypeError:
            scores = self.inner.predict(pairs)
        return np.asarray(scores, dtype=np.float32)


# ============================================================================
# 모델 로딩/라우팅 (20종)
# ============================================================================
def load_model(model_name: str, device: str, batch_size: int = 16):
    """mteb.evaluate 에 넘길 모델 반환.
    - 표준 ST CrossEncoder → CrossEncoder 그대로(mteb 가 CrossEncoderWrapper 로 래핑).
    - 비-ST → 해당 wrapper + MtebRerankAdapter.
    """
    def adapt(w):
        a = MtebRerankAdapter(w, model_name, batch_size=batch_size)
        # 실제 적용된 max_length 기록(재현성). wrapper 별 속성명이 다르므로 순차 탐색.
        eff = (getattr(w, "max_length", None)
               or getattr(w, "max_doc_tokens", None)
               or getattr(getattr(w, "model", None), "max_length", None)
               or MAX_LENGTH)
        a.eval_max_length = eff
        return a
    from sentence_transformers import CrossEncoder
    from wrappers import (
        Qwen3RerankerWrapper,
        QwenSeqClsWrapper,
        MxbaiRerankerWrapper,
        BGEGemmaRerankerWrapper,
        JinaRerankerV3Wrapper,
        NemotronRerankerWrapper,
    )

    name = model_name.lower()
    bf16 = {"dtype": torch.bfloat16}

    # Qwen3-Reranker seq-cls → native seq-cls wrapper (ST5.7 chat_template 회피, 4B/8B 대응)
    if "qwen3-reranker" in name:
        w = QwenSeqClsWrapper(model_name, device=device, torch_dtype=torch.bfloat16, max_length=MAX_LENGTH)
        return adapt(w)
    # PIXIE (Qwen 프롬프트, ST CrossEncoder 로 정상 동작)
    if "pixie-spell-reranker" in name:
        w = Qwen3RerankerWrapper(model_name, trust_remote_code=True, model_kwargs=bf16, device=device)
        w.model.max_length = min(MAX_LENGTH, getattr(w.model, "max_length", MAX_LENGTH))  # cap
        return adapt(w)
    if "mxbai" in name:
        w = MxbaiRerankerWrapper(model_name, device=device, torch_dtype=torch.bfloat16)
        try:
            w.model.max_length = min(MAX_LENGTH, getattr(w.model, "max_length", MAX_LENGTH))  # cap
        except Exception:
            pass
        return adapt(w)
    if "bge-reranker-v2-gemma" in name:
        w = BGEGemmaRerankerWrapper(model_name, use_bf16=True, devices=[device], max_length=MAX_LENGTH)
        return adapt(w)
    if "jina-reranker-v3" in name:  # v3, v3.5
        w = JinaRerankerV3Wrapper(model_name, device=device, torch_dtype=torch.bfloat16, max_doc_tokens=MAX_LENGTH)
        return adapt(w)
    if "nemotron-rerank" in name:
        w = NemotronRerankerWrapper(model_name, device=device, torch_dtype=torch.bfloat16, max_length=MAX_LENGTH)
        return adapt(w)

    # 표준 ST CrossEncoder (bge, ettin, zerank-2, LightOn, LAMAR, gte, jina-v2, ko-reranker 계열 …)
    ce = CrossEncoder(model_name, trust_remote_code=True, model_kwargs=bf16, device=device)
    # ★ max_length = min(8192, 모델 네이티브 최대) 로 상한(cap). 8192 초과 모델(40960/262144 등)은 8192 로
    #   맞춰 공정 비교하되, 절대 위치 임베딩 모델(ko-reranker=XLM-R 514 등)에 8192 를 강제하면 position id
    #   오버플로 → CUDA device-side assert(컨텍스트 오염·연쇄 실패) 이므로 네이티브 한도를 넘기지 않는다.
    native = getattr(ce, "max_length", None) or MAX_LENGTH
    ce.max_length = min(MAX_LENGTH, native)
    ce.eval_max_length = ce.max_length
    tok = getattr(ce, "tokenizer", None)
    if tok is not None and getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token  # causal-LM 기반(zerank-2 등) batch>1 대응
    return ce


# ============================================================================
# task 로딩 + 점수 추출
# ============================================================================
def get_task(task_name: str):
    try:
        ts = mteb.get_tasks(tasks=[task_name], languages=KOR_LANG)
        if ts:
            return ts[0]
    except Exception:
        pass
    ts = mteb.get_tasks(tasks=[task_name])
    return ts[0] if ts else None


def extract_scores(res) -> dict:
    """ModelResult 를 walk 해 subset별 ndcg_at_{1,5,10}/map/mrr 수집 → subset 평균(=mteb 공식 집계)."""
    metrics = ["ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "map_at_10", "mrr_at_10", "recall_at_10"]
    found: dict[str, list[float]] = {m: [] for m in metrics}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in found and isinstance(v, (int, float)):
                    found[k].append(float(v))
                walk(v)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        else:
            for attr in ("scores", "task_results", "results"):
                if hasattr(o, attr):
                    try:
                        walk(getattr(o, attr))
                    except Exception:
                        pass

    walk(res)
    out = {}
    for m, vals in found.items():
        if vals:
            out[m] = sum(vals) / len(vals)
            out[m + "__per_subset"] = vals
    return out


# ============================================================================
# 드라이버
# ============================================================================
def eval_one_model(model_name: str, tasks: list[str], device: str, overwrite: bool, batch_size: int):
    model = load_model(model_name, device, batch_size)
    out_model_dir = OUT_DIR / model_name
    out_model_dir.mkdir(parents=True, exist_ok=True)
    for task_name in tasks:
        out_path = out_model_dir / f"{task_name}.json"
        if out_path.exists() and not overwrite:
            print(f"[skip] {model_name} / {task_name} (이미 있음)", flush=True)
            continue
        try:
            t = get_task(task_name)
            if t is None:
                print(f"[error] task 없음: {task_name}", flush=True)
                continue
            print(f"\n=== {model_name} × {task_name} ===", flush=True)
            res = mteb.evaluate(model, t, cache=None, overwrite_strategy="always", show_progress_bar=False, encode_kwargs={"batch_size": batch_size})
            raw = extract_scores(res)
            # simple 포맷(리더보드/README 공용): __per_subset 제외 + main_score·_split 추가.
            scores = {k: v for k, v in raw.items() if not k.endswith("__per_subset")}
            scores["main_score"] = scores.get("ndcg_at_10")
            scores["_model"] = model_name
            scores["_task"] = task_name
            scores["_split"] = "+".join(t.metadata.eval_splits)   # 예: MLDR = "dev+test" (공식 MTEB kor v2)
            scores["_max_length"] = getattr(model, "eval_max_length", MAX_LENGTH)  # 재현성: 실제 적용된 max_length(=min(8192,네이티브))
            scores["_neg_top_k"] = NEG_TOP_K        # 재현성: gold 외 BM25 negative 후보 수
            json.dump(scores, open(out_path, "w"), indent=2)
            print(f"[done] {task_name}: NDCG@10={scores.get('ndcg_at_10')}", flush=True)
        except Exception as ex:
            print(f"[error] {model_name} / {task_name}: {ex}", flush=True)
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="MTEB 2.x reranker 평가 (정답 후보 항상 포함)")
    ap.add_argument("--model_names", nargs="+", default=DEFAULT_MODEL_NAMES)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | models={len(args.model_names)} tasks={len(args.tasks)}", flush=True)
    for model_name in args.model_names:
        try:
            eval_one_model(model_name, args.tasks, device, args.overwrite, args.batch_size)
        except Exception as ex:
            print(f"[error] model {model_name}: {ex}", flush=True)
            traceback.print_exc()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
