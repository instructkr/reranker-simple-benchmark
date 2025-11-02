import streamlit as st
import os
import json
import pandas as pd

st.set_page_config(layout="wide")


def app():
    data = {}
    avg_data = {}
    tasks = [
        "Ko-StrategyQA",
        "AutoRAGRetrieval",
        "PublicHealthQA",
        "BelebeleRetrieval",
		"MIRACLRetrieval",
		"MrTidyRetrieval",
		"MultiLongDocRetrieval",
		"XPQARetrieval"
    ]
    top_k_types = ["top1", "top5", "top10"]

    score_types = {
        "top1": ["mrr_at_1", "map_at_1", "ndcg_at_1"],
        "top5": ["mrr_at_5", "map_at_5", "ndcg_at_5"],
        "top10": ["mrr_at_10", "map_at_10", "ndcg_at_10"],
    }

    for task in tasks:
        data[task] = {top_k: [] for top_k in top_k_types}

    root_dir = "results/stage2"

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            for task in tasks:
                if file == task + ".json":
                    with open(os.path.join(subdir, file)) as f:
                        d = json.load(f)
                        for top_k in top_k_types:
                            results = {}
                            for score in score_types[top_k]:
                                if "dev" in d["scores"] and "test" not in d["scores"]:
                                    results[score] = d["scores"]["dev"][0][score]
                                elif "test" in d["scores"] and "dev" not in d["scores"]:
                                    results[score] = d["scores"]["test"][0][score]
                                else:
                                    # dev, test를 모두 가지고 있는 평가 데이터셋을 위함
                                    results[score] = (d["scores"]["dev"][0][score] + d["scores"]["test"][0][score]) / 2

                            data[task][top_k].append(
                                (
                                    os.path.relpath(subdir, root_dir),
                                    results[score_types[top_k][0]],
                                    results[score_types[top_k][1]],
                                    results[score_types[top_k][2]],
                                )
                            )

    # 각 작업에 대해 top10, top1000 점수 표시
    for task in tasks:
        st.markdown(f"# {task}")
        for top_k in top_k_types:
            st.markdown(f"## {top_k.capitalize()} Scores")
            df = pd.DataFrame(
                data[task][top_k],
                columns=[
                    "Subdir",
                    f"MRR_{top_k}",
                    f"MAP_{top_k}",
                    f"NDCG_{top_k}",
                ],
            )
            df = df.sort_values(by=f"MRR_{top_k}", ascending=False)
            st.dataframe(df, use_container_width=True)

            # 각 모델의 평균 점수 계산
            for subdir, mrr, map_score, ndcg in data[task][top_k]:
                if subdir not in avg_data:
                    avg_data[subdir] = {
                        k: [[], [], []] for k in top_k_types
                    } 
                avg_data[subdir][top_k][0].append(mrr)
                avg_data[subdir][top_k][1].append(map_score)
                avg_data[subdir][top_k][2].append(ndcg)

    # 각 모델 별 평균 점수 계산 후 출력
    st.markdown("# Average Scores")
    for top_k in top_k_types:
        avg_results = []
        for model in avg_data:
            mrr_avg = (
                sum(avg_data[model][top_k][0]) / len(avg_data[model][top_k][0])
                if avg_data[model][top_k][0]
                else 0
            )
            map_avg = (
                sum(avg_data[model][top_k][1]) / len(avg_data[model][top_k][1])
                if avg_data[model][top_k][1]
                else 0
            )
            ndcg_avg = (
                sum(avg_data[model][top_k][2]) / len(avg_data[model][top_k][2])
                if avg_data[model][top_k][2]
                else 0
            )
            avg_results.append([model, mrr_avg, map_avg, ndcg_avg])

        avg_df = pd.DataFrame(
            avg_results,
            columns=[
                "Model",
                f"Average MRR_{top_k}",
                f"Average MAP_{top_k}",
                f"Average NDCG_{top_k}",
            ],
        )
        avg_df = avg_df.sort_values(by=f"Average MRR_{top_k}", ascending=False)
        st.markdown(f"## {top_k.capitalize()} Average Scores")
        st.dataframe(avg_df, use_container_width=True)


if __name__ == "__main__":
    app()