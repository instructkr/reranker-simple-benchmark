"""Korean Reranker 리더보드 (streamlit).

집계·표는 README(`build_readme_mteb2x.py`)와 **동일한 소스·방식**을 재사용한다:
  - 결과: `eval/results/stage2/<model>/<task>.json` (task별 mteb 공식 get_score 값 = subset·split 평균,
    예: MultiLongDocRetrieval = dev+test 평균).
  - 표: 공식 9-subset mean + MLDR 제외 8-subset mean(listwise/long-doc OOD 공정 비교) + per-dataset NDCG@10.
지표는 NDCG@1/5/10 중심. 정답(gold)을 재랭킹 후보에 항상 포함, max_length 8192(ko-reranker 512, ettin 7999).

실행: cd eval && uv run streamlit run leaderboard_reranker.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_readme_mteb2x import collect, _mean_over, size_label, TASKS, MLDR  # noqa: E402

st.set_page_config(layout="wide")


def _mean_df(data, tasks):
    rows = []
    for model in data:
        r = _mean_over(data, model, tasks)  # tasks 전부 있어야 값(하나라도 결측이면 제외)
        if r:
            rows.append([model, size_label(model), round(r[0], 4), round(r[1], 4), round(r[2], 4)])
    df = pd.DataFrame(rows, columns=["Model", "Params", "Mean NDCG@1", "Mean NDCG@5", "Mean NDCG@10"])
    return df.sort_values("Mean NDCG@10", ascending=False).reset_index(drop=True)


def app():
    data = collect()
    tasks_8 = [t for t in TASKS if t != MLDR]

    st.title("Korean Reranker Leaderboard — gold-injected, official kMTEB (mteb 2.x)")
    st.caption(
        "정답(gold) 문서를 재랭킹 후보에 항상 포함(BM25 top-50 ∪ gold). max_length 8192 "
        "(ko-reranker 512, ettin 7999). 집계 = mteb 공식 get_score(subset·split 평균, MLDR=dev+test)."
    )

    st.header("Official kMTEB (9 subsets)")
    st.caption("9개 공식 subset 을 모두 평가한 모델. listwise 모델(장문 미완)은 아래 8-subset 표 참고.")
    st.dataframe(_mean_df(data, TASKS), use_container_width=True)

    st.header("MLDR 제외 (8 subsets · listwise / long-doc OOD 공정 비교)")
    st.caption(
        "장문(MultiLongDocRetrieval) 제외 — listwise 모델(jina-reranker-v3/v3.5)의 token-length OOD 공정 비교. "
        "MLDR 제외로 전 모델 mean 이 9-subset 대비 상승 → 표 간 절대값 비교 금지."
    )
    st.dataframe(_mean_df(data, tasks_8), use_container_width=True)

    st.header("Per-dataset NDCG@10")
    order = sorted(
        data.keys(),
        key=lambda m: (_mean_over(data, m, TASKS) or _mean_over(data, m, tasks_8) or (0, 0, 0))[2],
        reverse=True,
    )
    rows = []
    for model in order:
        row = {"Model": model, "Params": size_label(model)}
        for t in TASKS:
            v = data[model].get(t, {}).get("ndcg_at_10")
            row[t] = round(v, 4) if v is not None else None
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    app()
