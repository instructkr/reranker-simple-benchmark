"""모델 크기(파라미터 수) vs 성능(공식 MTEB(kor, v2) 9-subset mean NDCG@10) 산점도.

- x축: 파라미터 수 (log scale, MODEL_SIZES 실측)
- y축: 9개 공식 subset 평균 NDCG@10 (9개 전부 완료한 모델만 → listwise jina-reranker-v3/v3.5 는
        MLDR token-length OOD 로 자동 제외)
- 단일 계열(모든 reranker 동일 카테고리) → 단일 색·범례 없음. 각 점에 모델명 직접 라벨.

matplotlib 은 코어 의존성이 아니므로 임시 주입으로 실행:
  uv run --with matplotlib python eval/plot_size_vs_ndcg.py
산출물: assets/model_size_vs_ndcg9.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

from build_readme_mteb2x import collect, MODEL_SIZES, _mean_over, TASKS

V2_ROOT = Path(__file__).resolve().parents[1]
OUT = V2_ROOT / "assets" / "model_size_vs_ndcg9.png"

INK = "#1a2027"
MUTED = "#8a94a6"
DOT = "#2166ac"          # 단일 계열 색 (접근성 양호한 강한 파랑)


# 라벨용 축약명 (충돌·오버플로 완화). 없으면 basename.
SHORT = {
    "tomaarsen/Qwen3-Reranker-8B-seq-cls": "Qwen3-8B",
    "tomaarsen/Qwen3-Reranker-4B-seq-cls": "Qwen3-4B",
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": "Qwen3-0.6B",
    "jinaai/jina-reranker-v3": "jina-v3",
    "zeroentropy/zerank-2-reranker": "zerank-2",
    "lightonai/LightOn-rerank-PW-4B": "LightOn-4B",
    "mixedbread-ai/mxbai-rerank-large-v2": "mxbai-large-v2",
    "BAAI/bge-reranker-v2-m3": "bge-v2-m3",
    "nvidia/llama-nemotron-rerank-1b-v2": "nemotron-1b",
    "nlpai-lab/LAMAR-600m": "LAMAR-600m",
    "dragonkue/bge-reranker-v2-m3-ko": "bge-v2-m3-ko",
    "BAAI/bge-reranker-v2-gemma": "bge-v2-gemma",
    "upskyy/ko-reranker-8k": "ko-reranker-8k",
    "Dongjin-kr/ko-reranker": "ko-reranker",
    "telepix/PIXIE-Spell-Reranker-Preview-0.6B": "PIXIE-0.6B",
    "cross-encoder/ettin-reranker-1b-v1": "ettin-1b",
}


def short(model):
    return SHORT.get(model, model.split("/")[-1])


def main():
    data = collect()
    pts = []
    for model in data:
        r = _mean_over(data, model, TASKS)   # 9개 전부 있어야 값 → v3.5 자동 제외
        if r and model in MODEL_SIZES:
            pts.append((MODEL_SIZES[model] / 1e9, r[2], short(model)))
    pts.sort()

    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=150)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, s=110, color=DOT, edgecolor="white", linewidth=1.6, zorder=3)

    # 우측(≈1B+): 점 오른쪽에 직접 라벨. 좌측 밀집(≈0.6B): 오른쪽 빈 공간에 사다리 배치+연결선.
    for x, y, name in pts:
        if x > 0.65:
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(10, 0),
                        ha="left", va="center", fontsize=8.4, color=INK)

    left = sorted([p for p in pts if p[0] <= 0.65], key=lambda p: -p[1])  # y 내림차순
    n = len(left)
    ytop = max(p[1] for p in left) + 0.016
    ybot = min(p[1] for p in left) - 0.016
    ladder = [ytop - i * (ytop - ybot) / (n - 1) for i in range(n)]
    for (x, y, name), yl in zip(left, ladder):
        ax.annotate(name, xy=(x, y), xytext=(0.70, yl), textcoords="data",
                    ha="left", va="center", fontsize=8.0, color=INK,
                    arrowprops=dict(arrowstyle="-", color="#c3c9d4", lw=0.6,
                                    shrinkA=1, shrinkB=5))

    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 4, 8]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}B" for t in ticks]))
    ax.set_xlim(0.45, 11)
    ax.set_ylim(min(ys) - 0.02, max(ys) + 0.02)

    ax.set_xlabel("Model size (parameters, log scale)", fontsize=11, color=INK)
    ax.set_ylabel("Mean NDCG@10  (official MTEB(kor, v2), 9 subsets)", fontsize=11, color=INK)
    ax.set_title("Reranker model size vs. accuracy  —  gold-injected reranking, max_length 8192\n"
                 "(jina-reranker-v3/v3.5 excluded: listwise long-doc token-length OOD)",
                 fontsize=11.5, color=INK, pad=12)

    ax.grid(True, which="major", color="#e6e9ef", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT} ({len(pts)} models)")


if __name__ == "__main__":
    main()
