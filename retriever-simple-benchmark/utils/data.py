import pandas as pd


def load_data(qa_parquet_path: str, corpus_parquet_path: str) -> tuple:
    """
    Load and preprocess data from the given parquet files.
    :param qa_parquet_path: Path to QA parquet file.
    :param corpus_parquet_path: Path to corpus parquet file.
    :return: qd_df, corpus_df, valid_dict
    """

    qd_df = pd.read_parquet(qa_parquet_path)
    # Extract the doc_id from nested lists: e.g. [[doc_id]] -> doc_id
    qd_df["retrieval_gt"] = qd_df["retrieval_gt"].apply(lambda x: x[0][0])

    corpus_df = pd.read_parquet(corpus_parquet_path)
    corpus_df = corpus_df[["doc_id", "contents"]].reset_index(drop=True)
    qd_df = qd_df[["qid", "query", "generation_gt", "retrieval_gt"]].reset_index(
        drop=True
    )

    # Build qrel dictionary
    qrel_id = {}
    for _, row in qd_df[["qid", "retrieval_gt"]].iterrows():
        q_id, doc_id = row
        if q_id not in qrel_id:
            qrel_id[q_id] = set()
        qrel_id[q_id].add(doc_id)

    valid_dict = {"qrel": qrel_id}

    return qd_df, corpus_df, valid_dict
