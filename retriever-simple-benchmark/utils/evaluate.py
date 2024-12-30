import time
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
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
    Evaluate the reranker on the entire corpus/query data, splitting queries across processes
    if running under torchrun. Then gather partial results on rank 0, compute global metrics, and
    broadcast them to all ranks. TQDM progress bar is shown only on rank 0 (for readability).

    Steps:
      1) Split qd_df among processes (rank).
      2) Each rank does local inference => local_ranks_list, local_scores_list, local_qids_list
         with a progress bar (only rank=0).
      3) Gather all local data on rank 0
      4) Rank 0 merges them in the correct global order, runs calculate_accuracy/f1/etc.
      5) Rank 0 broadcasts the final metrics to other ranks
      6) Return the metrics so all ranks see the same result

    :param corpus_df: DataFrame [doc_id, contents].
    :param qd_df: DataFrame [qid, query, retrieval_gt, ...].
    :param valid_dict: dict with 'qrel' => { qid: set(doc_ids) }.
    :param reranker: Reranker with `compute_score_batch(query, doc_texts, normalize=...)`.
    :param batch_size: batch size for inference on (query, doc) pairs.
    :return: (accuracies, f1_scores, recalls, precisions, total_inference_time, avg_inference_time).
    """

    # ---------------------------
    # 1) Check if distributed
    # ---------------------------
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_ranks_list: list[np.ndarray] = []
    local_scores_list: list[np.ndarray] = []
    local_qids_list: list = []  # store the QIDs in local order
    inference_times = []

    doc_texts = corpus_df["contents"].tolist()

    # ---------------------------
    # 2) Split qd_df among ranks
    # ---------------------------
    num_queries = len(qd_df)
    chunk_size = num_queries // world_size
    remainder = num_queries % world_size

    start = rank * chunk_size
    end = start + chunk_size
    if rank < remainder:
        start += rank
        end += rank + 1
    else:
        start += remainder
        end += remainder

    local_qd_df = qd_df.iloc[start:end]

    # If no queries for this rank, skip computations
    if len(local_qd_df) == 0:
        total_inference_time = 0.0
        avg_inference_time = 0.0
    else:
        # ---------------------------
        # 3) Local Inference w/ tqdm
        # ---------------------------
        # We'll wrap `enumerate(local_qd_df.itertuples(...))` with tqdm
        # Only rank=0 shows the bar (disable=(rank!=0))
        pbar = tqdm(
            enumerate(local_qd_df.itertuples(index=False)),
            desc=f"[Rank {rank}] Inference",
            total=len(local_qd_df),
            disable=(rank != 0),
        )
        for i, row in pbar:
            qid = row.qid
            query_str = row.query

            t0 = time.time()

            scores = batched_compute_score(
                reranker=reranker,
                query=query_str,
                doc_texts=doc_texts,
                batch_size=batch_size,
            )

            scores_arr = np.array(scores, dtype=np.float32)
            sorted_idxs = np.argsort(-scores_arr)

            local_scores_list.append(scores_arr[sorted_idxs])
            local_ranks_list.append(sorted_idxs)
            local_qids_list.append(qid)

            t1 = time.time()
            elapsed = t1 - t0
            inference_times.append(elapsed)

        total_inference_time = sum(inference_times)
        avg_inference_time = total_inference_time / len(local_qd_df)

    # ---------------------------
    # 4) Gather local data on rank 0
    # ---------------------------
    local_result = {
        "qids": local_qids_list,
        "ranks_list": local_ranks_list,
        "scores_list": local_scores_list,
        "times": inference_times,
    }

    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, local_result)

    local_times_info = (total_inference_time, avg_inference_time)
    gather_list_times = [None] * world_size
    dist.all_gather_object(gather_list_times, local_times_info)

    # ---------------------------
    # 5) On rank 0, merge + compute global metrics
    # ---------------------------
    import pickle

    if rank == 0:
        # Merge partial results
        global_map = {}

        for r in range(world_size):
            qids_r = gather_list[r]["qids"]
            ranks_r = gather_list[r]["ranks_list"]
            scores_r = gather_list[r]["scores_list"]

            for qid_, ranks_arr, scores_arr in zip(qids_r, ranks_r, scores_r):
                global_map[qid_] = (ranks_arr, scores_arr)

        # Rebuild final_ranks_list in qd_df order
        final_ranks_list = []
        final_scores_list = []
        for row in qd_df.itertuples(index=False):
            qid = row.qid
            ranks_, scores_ = global_map[qid]
            final_ranks_list.append(ranks_)
            final_scores_list.append(scores_)

        # Compute global metrics
        k_values = [1, 3, 5, 10]
        accuracies = calculate_accuracy(
            final_ranks_list, valid_dict, qd_df, corpus_df, k_values
        )
        f1_scores, recalls, precisions = calculate_f1_recall_precision(
            final_ranks_list, valid_dict, qd_df, corpus_df, k_values
        )

        # Summation of total_inference_time
        sum_total = 0.0
        for ti, _ in gather_list_times:
            sum_total += ti

        global_total_inference_time = sum_total
        global_avg_inference_time = (
            global_total_inference_time / len(qd_df) if len(qd_df) else 0.0
        )

        final_metrics = {
            "acc": accuracies,
            "f1": f1_scores,
            "recall": recalls,
            "precision": precisions,
            "total_time": global_total_inference_time,
            "avg_time": global_avg_inference_time,
        }

        metrics_bytes = pickle.dumps(final_metrics)
    else:
        metrics_bytes = b""

    # broadcast pickled metrics from rank0
    length_tensor = torch.tensor(
        [len(metrics_bytes)],
        dtype=torch.long,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dist.broadcast(length_tensor, src=0)
    length_val = length_tensor.item()

    recv_buffer = torch.empty(
        [length_val], dtype=torch.uint8, device=length_tensor.device
    )
    if rank == 0:
        recv_buffer[:length_val] = torch.tensor(
            list(metrics_bytes), dtype=torch.uint8, device=length_tensor.device
        )
    dist.broadcast(recv_buffer, src=0)
    if rank != 0:
        metrics_bytes = bytes(recv_buffer.tolist())

    final_metrics_unpacked = pickle.loads(metrics_bytes)

    # Return them in the usual tuple format
    accuracies = final_metrics_unpacked["acc"]
    f1_scores = final_metrics_unpacked["f1"]
    recalls = final_metrics_unpacked["recall"]
    precisions = final_metrics_unpacked["precision"]
    total_time = final_metrics_unpacked["total_time"]
    avg_time = final_metrics_unpacked["avg_time"]

    return (accuracies, f1_scores, recalls, precisions, total_time, avg_time)


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
