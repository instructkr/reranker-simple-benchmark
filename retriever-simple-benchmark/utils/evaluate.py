import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


def evaluate_model(
    corpus_df: pd.DataFrame,
    qd_df: pd.DataFrame,
    valid_dict: dict,
    reranker,
    batch_size: int = 32,
) -> tuple[
    dict[int, float],  # accuracies
    dict[int, float],  # f1_scores
    dict[int, float],  # recalls
    dict[int, float],  # precisions
    float,  # total_inference_time
    float,  # avg_inference_time
]:
    """
    Single-process evaluation of a reranker on the entire corpus/query data.

    :param corpus_df: DataFrame [doc_id, contents]
    :param qd_df:     DataFrame [qid, query, retrieval_gt, ...]
    :param valid_dict: dict with 'qrel' => { qid: set(doc_ids) }.
    :param reranker:  Reranker with `compute_score_batch(query, doc_texts, normalize=...)`
    :param batch_size: batch size for inference on (query, doc) pairs
    :return: (accuracies, f1_scores, recalls, precisions, total_inference_time, avg_inference_time).
    """

    # Prepare
    doc_texts = corpus_df["contents"].tolist()
    ranks_list = []
    scores_list = []
    inference_times = []

    # Loop over queries (single process, single rank).
    for row in tqdm(qd_df.itertuples(index=False), desc="Inference", total=len(qd_df)):
        _ = row.qid
        query_str = row.query

        t0 = time.time()

        # Compute scores for entire corpus with batching
        scores = batched_compute_score(reranker, query_str, doc_texts, batch_size)
        scores_arr = np.array(scores, dtype=np.float32)

        # Sort descending by score
        sorted_idxs = np.argsort(-scores_arr)
        ranks_list.append(sorted_idxs)
        scores_list.append(scores_arr[sorted_idxs])

        t1 = time.time()
        inference_times.append(t1 - t0)

    # Compute total/average inference times
    total_inference_time = sum(inference_times)
    avg_inference_time = total_inference_time / len(qd_df) if len(qd_df) else 0.0

    # Compute final metrics
    k_values = [1, 3, 5, 10]
    accuracies = calculate_accuracy(ranks_list, valid_dict, qd_df, corpus_df, k_values)
    f1_scores, recalls, precisions = calculate_f1_recall_precision(
        ranks_list, valid_dict, qd_df, corpus_df, k_values
    )

    return (
        accuracies,
        f1_scores,
        recalls,
        precisions,
        total_inference_time,
        avg_inference_time,
    )


def batched_compute_score(
    reranker, query: str, doc_texts: list[str], batch_size: int
) -> list[float]:
    scores = []
    total_docs = len(doc_texts)
    start_idx = 0
    while start_idx < total_docs:
        end_idx = min(start_idx + batch_size, total_docs)
        batch_docs = doc_texts[start_idx:end_idx]
        batch_scores = reranker.compute_score_batch(query, batch_docs, normalize=False)
        scores.extend(batch_scores)
        start_idx = end_idx

    # optional min-max normalization
    arr = np.array(scores, dtype=np.float32)
    min_s = float(arr.min())
    max_s = float(arr.max())
    if max_s - min_s > 1e-8:
        arr = (arr - min_s) / (max_s - min_s)
    else:
        arr[:] = 0.0

    return arr.tolist()


def calculate_accuracy(
    ranks_list, valid_dict, qd_df, corpus_df, k_values=None
) -> dict[int, float]:
    if k_values is None:
        k_values = [1, 3, 5]
    accuracies = {k: 0 for k in k_values}
    total_queries = len(qd_df)

    for i in range(total_queries):
        qid = qd_df.iloc[i]["qid"]
        search_idx = ranks_list[i]
        true_doc_id = list(valid_dict["qrel"][qid])[0]
        true_doc_idx = corpus_df[corpus_df["doc_id"] == true_doc_id].index[0]

        for k in k_values:
            top_k_preds = search_idx[:k]
            if true_doc_idx in top_k_preds:
                accuracies[k] += 1

    return {k: accuracies[k] / total_queries for k in k_values}


def calculate_f1_recall_precision(
    ranks_list, valid_dict, qd_df, corpus_df, k_values=None
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    if k_values is None:
        k_values = [1, 3, 5]
    f1_scores = {k: 0 for k in k_values}
    recall_scores = {k: 0 for k in k_values}
    precision_scores = {k: 0 for k in k_values}

    total_queries = len(qd_df)
    for i in range(total_queries):
        qid = qd_df.iloc[i]["qid"]
        search_idx = ranks_list[i]
        true_doc_id = list(valid_dict["qrel"][qid])[0]
        true_doc_idx = corpus_df[corpus_df["doc_id"] == true_doc_id].index[0]

        for k in k_values:
            top_k_preds = search_idx[:k]
            y_true = [1 if idx == true_doc_idx else 0 for idx in top_k_preds]
            y_pred = [1] * len(top_k_preds)

            precision_scores[k] += precision_score(y_true, y_pred, zero_division=0)
            recall_scores[k] += recall_score(y_true, y_pred, zero_division=0)
            f1_scores[k] += f1_score(y_true, y_pred, zero_division=0)

    # average across total queries
    return (
        {k: f1_scores[k] / total_queries for k in k_values},
        {k: recall_scores[k] / total_queries for k in k_values},
        {k: precision_scores[k] / total_queries for k in k_values},
    )
