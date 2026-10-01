"""추론 처리량(9-subset mean PPS) vs 성능(공식 MTEB(kor, v2) 9-subset mean NDCG@10) 산점도.

- x축: 9개 공식 subset 평균 PPS (구간별 로그 배율: <20 ×0.5, 20–26 ×4, 26–50 ×0.6, >50 ×1)
- y축: 9개 공식 subset 평균 NDCG@10 (9개 전부 완료한 모델만 → jina-reranker-v3/v3.5 자동 제외)
- 점 색: 모델 크기 구간(<0.5B / 0.5–1B / 1–3B / ≥3B), 라벨: 축약명 + 작은 글씨로 파라미터 수(B, 소수 첫째 자리)

matplotlib 은 코어 의존성이 아니므로 임시 주입으로 실행:
  uv run --with matplotlib python eval/plot_pps_vs_ndcg.py
산출물: assets/pps_vs_ndcg9.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

from stage2_results import collect, mean_pps, _mean_over, MODEL_SIZES, TASKS

V2_ROOT = Path(__file__).resolve().parents[1]
OUT = V2_ROOT / "assets" / "pps_vs_ndcg9.png"

INK = "#1a2027"
MUTED = "#8a94a6"
# Times New Roman 계열 — 미설치 환경은 metric 호환 대체 폰트(TeX Gyre Termes, Nimbus Roman) 로 fallback.
TIMES = ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"]

# 모델 크기 구간별 색 (상한 B, 범례명, 색). 저채도 4색, all-pairs CVD·일반 시각 구분 검증 통과.
SIZE_BINS = [
    (0.5, "< 0.5B", "#b8646e"),
    (1.0, "0.5–1B", "#6f9fc2"),
    (3.0, "1–3B", "#d6a55c"),
    (float("inf"), "≥ 3B", "#4d4a7a"),
]

# 라벨용 축약명 (충돌·오버플로 완화). 없으면 basename.
SHORT = {
    "Qwen/Qwen3-Reranker-8B": "Qwen3-8B",
    "Qwen/Qwen3-Reranker-4B": "Qwen3-4B",
    "Qwen/Qwen3-Reranker-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-Reranker-V1-Large-R2": "KaLM-Large-R2",
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
    "nlpai-lab/KURE-Reranker-nano": "KURE-Reranker-nano",
    "nlpai-lab/KURE-Reranker-base": "KURE-Reranker-base",
}

# 점이 겹치는 모델의 라벨 오프셋(points). 없으면 점 오른쪽. dx<0 이면 점 왼쪽에 우측 정렬.
OFFSET = {
    "zerank-2": (-16, -15, "center"),  # 점 바로 아래
    "Qwen3-8B": (10, 5),
    "KaLM-Large-R2": (10, 0),
    "Qwen3-4B": (-14, -14, "center"),  # 점 바로 아래
    "mxbai-large-v2": (10, 6),
    "LightOn-4B": (10, -5),
    "Qwen3-0.6B": (10, 5),
    "PIXIE-0.6B": (10, -5),
    "bge-v2-m3": (10, 0),
    "LAMAR-600m": (10, 0),
    "bge-v2-m3-ko": (-10, 0),
    "ko-reranker-8k": (-10, 0),
}


def size_color(billions):
    return next(color for upper, _, color in SIZE_BINS if billions < upper)


def label(ax, x, y, name, size, dx, dy, align=None):
    """'name (size)' — size 는 작고 흐린 글씨. dx<0 이면 점 왼쪽에 우측 정렬, align="center" 면 이름을 dx 에 가운데 정렬."""
    if align == "center":
        name_t = ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                             ha="center", va="center", fontsize=12, color=INK)
        ax.annotate(f"({size})", xy=(1, 0), xycoords=name_t, textcoords="offset points", xytext=(3, 0),
                    ha="left", va="bottom", fontsize=9.5, color=MUTED)
        return
    left = dx < 0
    name_style = {"fontsize": 12, "color": INK}
    size_style = {"fontsize": 9.5, "color": MUTED}
    first_text, first_style = (f"({size})", size_style) if left else (name, name_style)
    second_text, second_style = (name, name_style) if left else (f"({size})", size_style)
    first = ax.annotate(first_text, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha="right" if left else "left", va="center", **first_style)
    ax.annotate(second_text, xy=(0 if left else 1, 0), xycoords=first,
                textcoords="offset points", xytext=(-3 if left else 3, 0),
                ha="right" if left else "left", va="bottom", **second_style)


def main():
    data = collect()
    pts = []
    for model in data:
        r = _mean_over(data, model, TASKS)
        pps = mean_pps(data, model, TASKS)
        if r and pps is not None:
            pts.append((pps, r[2], model))

    plt.rcParams.update({"font.family": "serif", "font.serif": TIMES, "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=150)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    for upper, name, color in SIZE_BINS:
        sel = [(x, y) for x, y, m in pts if size_color(MODEL_SIZES[m] / 1e9) == color]
        ax.scatter([x for x, _ in sel], [y for _, y in sel], s=110, color=color,
                   edgecolor="white", linewidth=1.6, zorder=3, label=name)

    for x, y, model in pts:
        name = SHORT.get(model, model.split("/")[-1])
        label(ax, x, y, name, f"{MODEL_SIZES[model] / 1e9:.1f}B", *OFFSET.get(name, (10, 0)))

    # 구간별 로그 배율: <20 ×0.5, 20–26 ×4 (Qwen3-4B·KaLM 분리), 26–50 ×0.6, >50 ×1
    A, B, C = np.log10([20, 26, 50])
    S0, S1, S2, S3 = 0.5, 4.0, 0.6, 1.0
    UB, UC = S1 * (B - A), S1 * (B - A) + S2 * (C - B)
    def fwd(x):
        t = np.log10(np.clip(x, 1e-9, None))
        return np.select([t < A, t < B, t < C], [S0 * (t - A), S1 * (t - A), UB + S2 * (t - B)], UC + S3 * (t - C))
    def inv(u):
        return 10 ** np.select([u < 0, u < UB, u < UC], [A + u / S0, A + u / S1, B + (u - UB) / S2], C + (u - UC) / S3)
    ax.set_xscale("function", functions=(fwd, inv))
    ticks = [10, 20, 25, 50, 100, 200, 500, 1000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(9, max(xs) * 2.3)
    ax.set_ylim(min(ys) - 0.02, max(ys) + 0.03)  # 위쪽 여유: 그래프 안 범례 자리

    ax.set_xlabel("Mean PPS (pairs/s)", fontsize=14, color=INK)
    ax.set_ylabel("Mean nDCG@10", fontsize=14, color=INK)
    # 범례: 그래프 안 위쪽, 오른쪽 끝을 x=900 에 맞춤 (1000 눈금 앞), "Model Size" 제목까지 한 상자로 묶어 가로 한 줄
    items = [TextArea("Model Size", textprops={"fontsize": 11, "color": INK})]
    for _, name, color in SIZE_BINS:
        dot = DrawingArea(11, 11)
        dot.add_artist(Circle((5.5, 5.5), 5, facecolor=color, edgecolor="white", linewidth=1.0))
        items.append(HPacker(children=[dot, TextArea(name, textprops={"fontsize": 11, "color": INK})],
                             pad=0, sep=4, align="center"))
    box = AnchoredOffsetbox(loc="upper right", child=HPacker(children=items, pad=0, sep=16, align="center"),
                            bbox_to_anchor=(900, 0.985), bbox_transform=ax.get_xaxis_transform(), frameon=True,
                            pad=0.45, borderpad=0)
    box.patch.set_boxstyle("round,pad=0.2,rounding_size=0.4")
    box.patch.set_edgecolor("#d5dae2"); box.patch.set_linewidth(0.8); box.patch.set_facecolor("white")
    ax.add_artist(box)

    ax.grid(True, which="major", color="#e6e9ef", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=12)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT} ({len(pts)} models)")


if __name__ == "__main__":
    main()
