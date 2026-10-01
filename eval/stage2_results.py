"""stage2 결과(eval/results/stage2/<model>/<task>.json) 로딩·집계 — streamlit 리더보드와 산점도가 공용.

`collect()` 는 {model: {task: {ndcg_at_1, ndcg_at_5, ndcg_at_10, pps}}} 를, `_mean_over` 는 task 평균 NDCG@1/5/10 을,
`mean_pps` 는 MLDR 제외 8-task 평균 PPS 를 돌려준다. `MODEL_SIZES` 는 모델 파라미터 수(실측).
"""
import glob
import json
import os
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
STAGE2 = V2_ROOT / "eval/results/stage2"

TASKS = [
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


def collect():
    """{model: {task: {ndcg_at_1, ndcg_at_5, ndcg_at_10, pps}}}"""
    data = {}
    for f in glob.glob(str(STAGE2 / "**" / "*.json"), recursive=True):
        base = os.path.basename(f)[:-5]
        if base not in TASKS:
            continue
        model = os.path.relpath(f, STAGE2).rsplit("/" + base + ".json", 1)[0]
        d = json.load(open(f))
        data.setdefault(model, {})[base] = {
            k: d.get(k) for k in ("ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "pps")
        }
    return data


MLDR = "MultiLongDocRetrieval"  # 장문 task — listwise 모델은 token-length OOD 로 8192 tractable 초과
# 8-task PPS 는 MLDR 제외 8 subset 평균 — 장문에선 모델별 입력 상한(512~8192)이 처리량을 지배하므로
# MLDR 은 평균에 넣지 않고 별도 열(MLDR PPS)로 둔다.
PPS_TASKS = [t for t in TASKS if t != MLDR]
PPS_NOTE = ("**8-task PPS** = 추론 처리량(query–document pairs/s), MLDR 제외 8 subset 평균. RTX A6000 1장, "
            "bf16 + flash_attention_2, `--speed` 로 측정 (측정 방식은 *Methodology* 참고).")

# 모델 파라미터 수(실측: 캐시된 safetensors 헤더의 tensor shape 합). 표 Params 열·산점도(크기 x축, PPS 그림 라벨·색)에 사용.
MODEL_SIZES = {
    "tomaarsen/Qwen3-Reranker-8B-seq-cls": 7_567_315_968,
    "tomaarsen/Qwen3-Reranker-4B-seq-cls": 4_021_787_136,
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": 595_777_536,
    "jinaai/jina-reranker-v3.5": 596_836_352,
    "jinaai/jina-reranker-v3": 596_836_352,
    "zeroentropy/zerank-2-reranker": 4_022_468_096,
    "lightonai/LightOn-rerank-PW-4B": 4_539_265_536,
    "mixedbread-ai/mxbai-rerank-large-v2": 1_543_714_304,
    "BAAI/bge-reranker-v2-m3": 567_755_777,
    "nvidia/llama-nemotron-rerank-1b-v2": 1_235_816_448,
    "nlpai-lab/LAMAR-600m": 567_755_777,
    "dragonkue/bge-reranker-v2-m3-ko": 567_755_777,
    "BAAI/bge-reranker-v2-gemma": 2_506_172_416,
    "upskyy/ko-reranker-8k": 567_755_777,
    "Dongjin-kr/ko-reranker": 559_891_457,
    "telepix/PIXIE-Spell-Reranker-Preview-0.6B": 595_777_536,
    "cross-encoder/ettin-reranker-1b-v1": 1_028_050_688,
    "nlpai-lab/KURE-Reranker-nano": 149_323_009,
    "nlpai-lab/KURE-Reranker-base": 1_720_574_976,
    "Qwen/Qwen3-Reranker-8B": 8_188_548_096,
    "Qwen/Qwen3-Reranker-4B": 4_021_784_576,
    "Qwen/Qwen3-Reranker-0.6B": 595_776_512,
    "KaLM-Embedding/KaLM-Reranker-V1-Large-R2": 7_508_928_880,
}


def size_label(model):
    p = MODEL_SIZES.get(model)
    if p is None:
        return "?"
    return f"{p / 1e9:.1f}B" if p >= 1e9 else f"{round(p / 1e6)}M"


def _mean_over(data, model, tasks):
    """model 이 tasks 전부에 유효 NDCG@10 이 있으면 (mean@1, mean@5, mean@10) 반환, 하나라도 결측이면 None."""
    v1 = v5 = v10 = 0.0
    for t in tasks:
        sc = data[model].get(t)
        if not sc or sc.get("ndcg_at_10") is None:
            return None
        v1 += sc.get("ndcg_at_1", 0.0)
        v5 += sc.get("ndcg_at_5", 0.0)
        v10 += sc["ndcg_at_10"]
    n = len(tasks)
    return v1 / n, v5 / n, v10 / n


def mean_pps(data, model, tasks=PPS_TASKS):
    """tasks(기본 MLDR 제외 8개) 전부에 PPS 가 있으면 평균, 하나라도 결측이면 None."""
    vals = [(data[model].get(t) or {}).get("pps") for t in tasks]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)
