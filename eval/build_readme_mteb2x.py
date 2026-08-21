"""stage2 결과 → README 표 3개 생성.

표1: 공식 kMTEB 9-subset mean NDCG@1/5/10 (NDCG@10 내림차순).
표2: MLDR 제외 8-subset mean NDCG@1/5/10 (listwise/long-doc OOD 공정 비교).
표3: 모델별 subset(dataset)별 NDCG@10.
각 표에 Params(모델 크기, 실측) 열 포함.

사용:
  uv run python eval/build_readme_mteb2x.py            # markdown 을 stdout 출력
  uv run python eval/build_readme_mteb2x.py --out tables.md
"""
import argparse
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
    """{model: {task: {ndcg_at_1, ndcg_at_5, ndcg_at_10}}}"""
    data = {}
    for f in glob.glob(str(STAGE2 / "**" / "*.json"), recursive=True):
        base = os.path.basename(f)[:-5]
        if base not in TASKS:
            continue
        model = os.path.relpath(f, STAGE2).rsplit("/" + base + ".json", 1)[0]
        d = json.load(open(f))
        data.setdefault(model, {})[base] = {
            k: d.get(k) for k in ("ndcg_at_1", "ndcg_at_5", "ndcg_at_10")
        }
    return data


MLDR = "MultiLongDocRetrieval"  # 장문 task — listwise 모델은 token-length OOD 로 8192 tractable 초과

# 모델 파라미터 수(실측: 캐시된 safetensors 헤더의 tensor shape 합). 표 Params 열·산점도 x축에 사용.
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


def _mean_table(data, tasks, title, note=None):
    rows = []
    for model in data:
        r = _mean_over(data, model, tasks)
        if r:
            rows.append((model,) + r)
    rows.sort(key=lambda x: x[3], reverse=True)
    out = [f"#### {title}", ""]
    if note:
        out += [note, ""]
    out.append("| Model | Params | Mean NDCG@1 | Mean NDCG@5 | Mean NDCG@10 |")
    out.append("|---|---|---|---|---|")
    for model, n1, n5, n10 in rows:
        out.append(f"| {model} | {size_label(model)} | {n1:.4f} | {n5:.4f} | {n10:.4f} |")
    out.append("")
    return out


def build_tables(data):
    tasks_8 = [t for t in TASKS if t != MLDR]
    out = []
    # 표1: 공식 kMTEB 9-subset (9개 전부 있는 모델만)
    out += _mean_table(
        data, TASKS,
        "Results — Official kMTEB (9 subsets)",
        "모든 9개 공식 subset 을 평가한 모델의 mean NDCG@k (NDCG@10 내림차순). "
        "listwise 모델처럼 장문(MLDR)을 완료하지 못한 모델은 아래 8-subset 표를 참고하세요.",
    )
    # 표2: MLDR 제외 8-subset (listwise/long-doc OOD 공정 비교 — 전 모델 공통 기준)
    out += _mean_table(
        data, tasks_8,
        "Results — MLDR 제외 (8 subsets, listwise / long-doc OOD 공정 비교)",
        "장문 검색(MultiLongDocRetrieval)은 listwise reranker 의 token-length OOD 로 8192 tractable "
        "시간 내 완료가 어렵습니다. 이를 제외한 8개 공통 subset 기준 mean NDCG@k (NDCG@10 내림차순) — "
        "모든 모델을 동일 기준으로 비교합니다.",
    )
    # 표3: subset별 NDCG@10 (9-subset 순서 유지, 전 모델)
    rows_all = sorted(
        data.keys(),
        key=lambda m: (_mean_over(data, m, TASKS) or _mean_over(data, m, tasks_8) or (0, 0, 0))[2],
        reverse=True,
    )
    out += ["#### Per-dataset NDCG@10", ""]
    out.append("| Model | Params | " + " | ".join(TASKS) + " |")
    out.append("|---|---|" + "---|" * len(TASKS))
    for model in rows_all:
        cells = [
            (f"{v:.4f}" if (v := data[model].get(t, {}).get("ndcg_at_10")) is not None else "—")
            for t in TASKS
        ]
        out.append(f"| {model} | {size_label(model)} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="-")
    args = ap.parse_args()
    data = collect()
    md = build_tables(data)
    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md)
        print(f"wrote {args.out} ({len(data)} models)")


if __name__ == "__main__":
    main()
