"""
Custom MTEB Task classes for SQuADKorV1Retrieval and WebFAQRetrieval
"""

from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
from datasets import load_dataset


class SQuADKorV1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SQuADKorV1Retrieval",
        description="Korean SQuAD v1.0 dataset for retrieval task",
        reference="https://huggingface.co/datasets/yjoonjang/squad_kor_v1",
        dataset={
            "path": "yjoonjang/squad_kor_v1",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 5774},
            "avg_character_length": {"test": {"average_document_length": 500, "average_query_length": 50}},
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # Load queries
        queries_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "queries",
            split="test",
            trust_remote_code=True
        )

        # Load corpus
        corpus_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "corpus",
            split="test",
            trust_remote_code=True
        )

        # Load qrels
        qrels_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "default",
            split="test",
            trust_remote_code=True
        )

        # Build queries dict: {split: {query_id: query_text}}
        self.queries = {"test": {}}
        for item in queries_ds:
            query_id = item["_id"]
            query_text = item["text"]
            self.queries["test"][query_id] = query_text

        # Build corpus dict: {split: {doc_id: {"title": title, "text": text}}}
        self.corpus = {"test": {}}
        for item in corpus_ds:
            doc_id = item["_id"]
            doc_title = item.get("title", "")
            doc_text = item["text"]
            self.corpus["test"][doc_id] = {
                "title": doc_title,
                "text": doc_text
            }

        # Build relevant_docs dict: {split: {query_id: {doc_id: score}}}
        self.relevant_docs = {"test": {}}
        for item in qrels_ds:
            query_id = item["query-id"]
            corpus_id = item["corpus-id"]
            score = item["score"]

            if query_id not in self.relevant_docs["test"]:
                self.relevant_docs["test"][query_id] = {}

            self.relevant_docs["test"][query_id][corpus_id] = score

        self.data_loaded = True


class WebFAQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WebFAQRetrieval",
        description="Korean Web FAQ dataset for retrieval task",
        reference="https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval",
        dataset={
            "path": "PaDaS-Lab/webfaq-retrieval",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {"test": {"average_document_length": 200, "average_query_length": 40}},
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # Load queries
        queries_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "kor-queries",
            split="test",
            trust_remote_code=True
        )

        # Load corpus
        corpus_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "kor-corpus",
            split="corpus",
            trust_remote_code=True
        )

        # Load qrels
        qrels_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "kor-qrels",
            split="test",
            trust_remote_code=True
        )

        # Build queries dict: {split: {query_id: query_text}}
        self.queries = {"test": {}}
        for item in queries_ds:
            query_id = item["_id"]
            query_text = item["text"]
            self.queries["test"][query_id] = query_text

        # Build corpus dict: {split: {doc_id: {"title": title, "text": text}}}
        self.corpus = {"test": {}}
        for item in corpus_ds:
            doc_id = item["_id"]
            doc_title = item.get("title", "")
            doc_text = item["text"]
            self.corpus["test"][doc_id] = {
                "title": doc_title,
                "text": doc_text
            }

        # Build relevant_docs dict: {split: {query_id: {doc_id: score}}}
        self.relevant_docs = {"test": {}}
        for item in qrels_ds:
            query_id = item["query-id"]
            corpus_id = item["corpus-id"]
            score = item["score"]

            if query_id not in self.relevant_docs["test"]:
                self.relevant_docs["test"][query_id] = {}

            self.relevant_docs["test"][query_id][corpus_id] = int(score)

        self.data_loaded = True
