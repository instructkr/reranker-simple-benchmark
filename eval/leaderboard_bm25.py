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
    ]
    top_k_types = ["top10", "top100", "top1000"]

    score_types = {
        "top10": ["recall_at_10", "precision_at_10", "ndcg_at_10"],
        "top100": ["recall_at_100", "precision_at_100", "ndcg_at_100"],
        "top1000": ["recall_at_1000", "precision_at_1000", "ndcg_at_1000"],
    }

    # 각 작업에 대한 데이터를 초기화
    for task in tasks:
        data[task] = {top_k: [] for top_k in top_k_types}

    root_dir = "results/stage1"

    # 데이터가 저장되어 있는 디렉토리의 모든 하위 폴더를 순회하면서 json 파일을 읽습니다.
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

                            # f1 score 직접 계산
                            f1_score = (
                                2 * (results[score_types[top_k][1]] * results[score_types[top_k][0]]) / (results[score_types[top_k][1]]+ results[score_types[top_k][0]])
                                if (results[score_types[top_k][1]]+ results[score_types[top_k][0]])> 0
                                else 0
                            )

                            data[task][top_k].append(
                                (
                                    os.path.relpath(subdir, root_dir),
                                    results[score_types[top_k][0]],
                                    results[score_types[top_k][1]],
                                    results[score_types[top_k][2]],
                                    f1_score,
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
                    f"Recall_{top_k}",
                    f"Precision_{top_k}",
                    f"NDCG_{top_k}",
                    f"F1_{top_k}",
                ],
            )
            df = df.sort_values(by=f"Recall_{top_k}", ascending=False)
            st.dataframe(df, use_container_width=True)

            # 각 모델의 평균 점수 계산
            for subdir, recall, precision, ndcg, f1 in data[task][top_k]:
                if subdir not in avg_data:
                    avg_data[subdir] = {
                        k: [[], [], [], []] for k in top_k_types
                    } 
                avg_data[subdir][top_k][0].append(recall)
                avg_data[subdir][top_k][1].append(precision)
                avg_data[subdir][top_k][2].append(ndcg)
                avg_data[subdir][top_k][3].append(f1)

    # 각 모델 별 평균 점수 계산 후 출력
    st.markdown("# Average Scores")
    for top_k in top_k_types:
        avg_results = []
        for model in avg_data:
            recall_avg = (
                sum(avg_data[model][top_k][0]) / len(avg_data[model][top_k][0])
                if avg_data[model][top_k][0]
                else 0
            )
            precision_avg = (
                sum(avg_data[model][top_k][1]) / len(avg_data[model][top_k][1])
                if avg_data[model][top_k][1]
                else 0
            )
            ndcg_avg = (
                sum(avg_data[model][top_k][2]) / len(avg_data[model][top_k][2])
                if avg_data[model][top_k][2]
                else 0
            )
            f1_avg = (
                sum(avg_data[model][top_k][3]) / len(avg_data[model][top_k][3])
                if avg_data[model][top_k][3]
                else 0
            )
            avg_results.append([model, recall_avg, precision_avg, ndcg_avg, f1_avg])

        avg_df = pd.DataFrame(
            avg_results,
            columns=[
                "Model",
                f"Average Recall_{top_k}",
                f"Average Precision_{top_k}",
                f"Average NDCG_{top_k}",
                f"Average F1_{top_k}",
            ],
        )
        avg_df = avg_df.sort_values(by=f"Average Recall_{top_k}", ascending=False)
        st.markdown(f"## {top_k.capitalize()} Average Scores")
        st.dataframe(avg_df, use_container_width=True)


if __name__ == "__main__":
    app()