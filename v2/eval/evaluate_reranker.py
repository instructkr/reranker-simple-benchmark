import os
import logging
from multiprocessing import Process, current_process
import torch
import json

import mteb
from mteb import MTEB
from sentence_transformers import CrossEncoder
from setproctitle import setproctitle
import traceback
from datasets import load_dataset

from mteb.tasks.Retrieval.multilingual.XPQARetrieval import XPQARetrieval
from mteb.tasks.Retrieval.multilingual.XPQARetrieval import _LANG_CONVERSION, _load_dataset_csv
from mteb.tasks.Retrieval.multilingual.BelebeleRetrieval import BelebeleRetrieval, _EVAL_SPLIT
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch

_original_load_results_file = DenseRetrievalExactSearch.load_results_file

def xpqa_load_data(self, **kwargs):
    if self.data_loaded:
        return

    path = self.metadata_dict["dataset"]["path"]
    revision = self.metadata_dict["dataset"]["revision"]
    eval_splits = self.metadata_dict["eval_splits"]
    dataset = _load_dataset_csv(path, revision, eval_splits)

    self.queries, self.corpus, self.relevant_docs = {}, {}, {}
    for lang_pair, _ in self.metadata.eval_langs.items():
        lang_corpus, lang_question = (
            lang_pair.split("-")[0],
            lang_pair.split("-")[1],
        )
        lang_not_english = lang_corpus if lang_corpus != "eng" else lang_question
        dataset_language = dataset.filter(
            lambda x: x["lang"] == _LANG_CONVERSION.get(lang_not_english)
        )
        question_key = "question_en" if lang_question == "eng" else "question"
        corpus_key = "candidate" if lang_corpus == "eng" else "answer"

        queries_to_ids = {
            eval_split: {
                q: f"Q{str(_id)}"
                for _id, q in enumerate(
                    sorted(set(dataset_language[eval_split][question_key])
                ))
            }
            for eval_split in eval_splits
        }

        self.queries[lang_pair] = {
            eval_split: {v: k for k, v in queries_to_ids[eval_split].items()}
            for eval_split in eval_splits
        }

        corpus_to_ids = {
            eval_split: {
                document: f"C{str(_id)}"
                for _id, document in enumerate(
                    sorted(set(dataset_language[eval_split][corpus_key])
                ))
            }
            for eval_split in eval_splits
        }

        self.corpus[lang_pair] = {
            eval_split: {
                v: {"text": k} for k, v in corpus_to_ids[eval_split].items()
            }
            for eval_split in eval_splits
        }

        self.relevant_docs[lang_pair] = {}
        for eval_split in eval_splits:
            self.relevant_docs[lang_pair][eval_split] = {}
            for example in dataset_language[eval_split]:
                query_id = queries_to_ids[eval_split].get(example[question_key])
                document_id = corpus_to_ids[eval_split].get(example[corpus_key])
                if query_id in self.relevant_docs[lang_pair][eval_split]:
                    self.relevant_docs[lang_pair][eval_split][query_id][
                        document_id
                    ] = 1
                else:
                    self.relevant_docs[lang_pair][eval_split][query_id] = {
                        document_id: 1
                    }

    self.data_loaded = True

def belebele_load_data(self, **kwargs) -> None:
    if self.data_loaded:
        return

    self.dataset = load_dataset(**self.metadata.dataset)

    self.queries = {lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets}
    self.corpus = {lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets}
    self.relevant_docs = {
        lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets
    }

    for lang_pair in self.hf_subsets:
        languages = self.metadata.eval_langs[lang_pair]
        lang_corpus, lang_question = (
            languages[0].replace("-", "_"),
            languages[1].replace("-", "_"),
        )
        ds_corpus = self.dataset[lang_corpus]
        ds_question = self.dataset[lang_question]

        question_ids = {
            question: _id
            for _id, question in enumerate(sorted(set(ds_question["question"])))
        }

        link_to_context_id = {}
        context_idx = 0
        for row in ds_corpus:
            if row["link"] not in link_to_context_id:
                context_id = f"C{context_idx}"
                link_to_context_id[row["link"]] = context_id
                self.corpus[lang_pair][_EVAL_SPLIT][context_id] = {
                    "title": "",
                    "text": row["flores_passage"],
                }
                context_idx = context_idx + 1

        for row in ds_question:
            query = row["question"]
            query_id = f"Q{question_ids[query]}"
            self.queries[lang_pair][_EVAL_SPLIT][query_id] = query

            context_link = row["link"]
            context_id = link_to_context_id[context_link]
            if query_id not in self.relevant_docs[lang_pair][_EVAL_SPLIT]:
                self.relevant_docs[lang_pair][_EVAL_SPLIT][query_id] = {}
            self.relevant_docs[lang_pair][_EVAL_SPLIT][query_id][context_id] = 1

    self.data_loaded = True

def patched_load_results_file(self):
    """JSONL 파일을 지원하는 패치된 load_results_file 메서드"""
    if self.previous_results.endswith('.jsonl'):
        previous_results = {}
        with open(self.previous_results, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = data["query_id"]
                relevance_ids = data["relevance_ids"]
                
                # 모든 문서에 동일한 점수 부여 (원래는 sorting을 위함)
                doc_scores = {doc_id: 1.0 for doc_id in relevance_ids}
                previous_results[query_id] = doc_scores
        
        assert isinstance(previous_results, dict)
        assert isinstance(previous_results[list(previous_results.keys())[0]], dict)
        return previous_results
    else:
        return _original_load_results_file(self)

BelebeleRetrieval.load_data = belebele_load_data
XPQARetrieval.load_data = xpqa_load_data
DenseRetrievalExactSearch.load_results_file = patched_load_results_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


# GPU별 task 매핑 - 필요에 따라 GPU 번호를 조정하세요
TASK_LIST_RERANKER_GPU_MAPPING = {
    0: [
        "Ko-StrategyQA",
        "AutoRAGRetrieval",
        "PublicHealthQA",
        "BelebeleRetrieval",
        "XPQARetrieval",
        "MultiLongDocRetrieval",
    ],
    1: ["MIRACLRetrieval"],
    2: ["MrTidyRetrieval"],
}

model_names = [
    "BAAI/bge-reranker-v2-m3",
    "dragonkue/bge-reranker-v2-m3-ko",
    "sigridjineth/ko-reranker-v1.1",
    "sigridjineth/ko-reranker-v1.2-preview",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "upskyy/ko-reranker-8k",
    "Dongjin-kr/ko-reranker",
    "jinaai/jina-reranker-v2-base-multilingual",
    # 여기에 다른 모델들 추가
]

previous_results_dir = "./results/stage1/top_1k_qrels"

def evaluate_reranker_model(model_name, gpu_id, tasks):
    try:
        device = torch.device(f"cuda:{str(gpu_id)}") 
        torch.cuda.set_device(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        setproctitle(f"{model_name}-reranker-{gpu_id}")
        print(f"Running tasks: {tasks} / {model_name} on GPU {gpu_id} in process {current_process().name}")
        
        cross_encoder = CrossEncoder(
            model_name, 
            trust_remote_code=True, 
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        
        output_dir = os.path.join("./results/stage2", model_name)
        
        # TODO 모델별 batch size 조정
        batch_size = 2048
        
        for task in tasks:
            print(f"Running task: {task} / {model_name} on GPU {gpu_id}")

            tasks_mteb = mteb.get_tasks(
                tasks=[task],
                languages=["kor-Kore", "kor-Hang", "kor_Hang"],
                eval_splits=["test"] if task == "MultiLongDocRetrieval" else None,
            )
            evaluation = MTEB(tasks=tasks_mteb)

            if os.path.exists(os.path.join(previous_results_dir, task + "_id.jsonl")):
                print(f"Previous results found: {task}")
                previous_results = os.path.join(previous_results_dir, task + "_id.jsonl")

                evaluation.run(
                    cross_encoder,
                    top_k=50,
                    save_predictions=True,
                    output_folder=output_dir,
                    previous_results=previous_results,
                    batch_size=batch_size
                )
            else:
                print(f"Previous results not found: {task}")
                evaluation.run(
                    cross_encoder,
                    top_k=50,
                    save_predictions=True,
                    output_folder=output_dir,
                    batch_size=batch_size
                )
                
    except Exception as ex:
        print(f"Error in GPU {gpu_id} with model {model_name}: {ex}")
        traceback.print_exc()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    for model_name in model_names:
        print(f"Starting evaluation for model: {model_name}")
        processes = []
        
        for gpu_id, tasks in TASK_LIST_RERANKER_GPU_MAPPING.items():
            p = Process(target=evaluate_reranker_model, args=(model_name, gpu_id, tasks))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        print(f"Completed evaluation for model: {model_name}")