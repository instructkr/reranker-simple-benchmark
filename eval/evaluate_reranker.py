"""
예시 script
uv run evaluate_reranker.py \
    --model_names tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
    --tasks Ko-StrategyQA AutoRAGRetrieval \
    --gpu_ids 0 1 \
    --batch_size 2 \
    --top_k 50 \
    --verbosity 1
"""


import os
import logging
from multiprocessing import Process, current_process, Queue
import torch
import json
import queue
from pathlib import Path
import argparse
from typing import List, Tuple

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

from wrappers import Qwen3RerankerWrapper, MxbaiRerankerWrapper, BGEGemmaRerankerWrapper

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
                
                # 역순으로 점수를 부여하여 원래 순서를 보존
                num_docs = len(relevance_ids)
                doc_scores = {doc_id: float(num_docs - i) for i, doc_id in enumerate(relevance_ids)}
                previous_results[query_id] = doc_scores
        
        assert isinstance(previous_results, dict)
        assert isinstance(previous_results[list(previous_results.keys())[0]], dict)
        return previous_results
    else:
        return _original_load_results_file(self)


BelebeleRetrieval.load_data = belebele_load_data
XPQARetrieval.load_data = xpqa_load_data
DenseRetrievalExactSearch.load_results_file = patched_load_results_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")


def evaluate_reranker_model(model_name: str, gpu_id: int, tasks: List[str], previous_results_dir: Path, output_base_dir: Path, top_k: int, verbosity: int, batch_size: int):
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        setproctitle(f"{model_name}-reranker-{gpu_id}")
        print(f"Running tasks: {tasks} / {model_name} on GPU {gpu_id} in process {current_process().name}")
        
        # Qwen 또는 PIXIE 모델인 경우 Qwen3RerankerWrapper 사용
        if "qwen" in model_name.lower() or "PIXIE-Spell-Reranker-Preview-0.6B" in model_name:
            print(f"Using Qwen3RerankerWrapper for {model_name}")
            model = Qwen3RerankerWrapper(
                model_name,
                trust_remote_code=True, 
                model_kwargs={"dtype": torch.bfloat16},
                device=device,
            )
        elif "mxbai" in model_name.lower():
            print(f"Using MxbaiRerankerWrapper for {model_name}")
            model = MxbaiRerankerWrapper(
                model_name,
                device=device,
                torch_dtype=torch.bfloat16,
            )
        elif "bge-reranker-v2-gemma" in model_name.lower():
            print(f"Using BGEGemmaRerankerWrapper for {model_name}")
            model = BGEGemmaRerankerWrapper(
                model_name, 
                use_bf16=True,
                devices=[device],
            )
        else:
            model = CrossEncoder(
                model_name, 
                trust_remote_code=True, 
                model_kwargs={"dtype": torch.bfloat16},
                device=device,
            )
        
        output_dir = output_base_dir / model_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for task in tasks:
            print(f"Running task: {task} / {model_name} on GPU {gpu_id}")

            tasks_mteb = mteb.get_tasks(
                tasks=[task],
                languages=["kor-Kore", "kor-Hang", "kor_Hang"],
                eval_splits=["test"] if task == "MultiLongDocRetrieval" else None,
            )
            evaluation = MTEB(tasks=tasks_mteb)

            previous_results_path = previous_results_dir / (task + "_id.jsonl")
            if previous_results_path.exists():
                print(f"Previous results found: {task}")
                previous_results = str(previous_results_path)

                evaluation.run(
                    model,
                    top_k=top_k,
                    save_predictions=False, # 이건 항상 False로 설정
                    output_folder=str(output_dir),
                    previous_results=previous_results,
                    encode_kwargs={"batch_size": batch_size},
                    verbosity=verbosity,
                )
            else:
                print(f"Previous results not found: {task}")
                evaluation.run(
                    model,
                    top_k=top_k,
                    save_predictions=False, # 이건 항상 False로 설정
                    output_folder=str(output_dir),
                    encode_kwargs={"batch_size": batch_size},
                    verbosity=verbosity,
                )
                
    except Exception as ex:
        print(f"Error in GPU {gpu_id} with model {model_name}: {ex}")
        traceback.print_exc()


def worker(job_queue: Queue, gpu_queue: Queue, previous_results_dir: Path, output_base_dir: Path, top_k: int, verbosity: int, batch_size: int):
    """작업 큐와 GPU 큐에서 작업을 가져와 실행하는 워커 함수"""
    while True:
        try:
            model_name, task = job_queue.get(timeout=1)
        except queue.Empty:
            break
        
        gpu_id = None
        try:
            gpu_id = gpu_queue.get()
            print(f"Process {current_process().name}: Starting task: {task} / {model_name} on GPU {gpu_id}")
            evaluate_reranker_model(model_name, gpu_id, [task], previous_results_dir, output_base_dir, top_k, verbosity, batch_size)
            print(f"Process {current_process().name}: Finished task: {task} / {model_name} on GPU {gpu_id}")
        except Exception:
            print(f"!!!!!!!!!! Process {current_process().name}: Error during task: {task} / {model_name} on GPU {gpu_id} !!!!!!!!!!!")
            traceback.print_exc()
        finally:
            if gpu_id is not None:
                gpu_queue.put(gpu_id)


# --- 기본 설정값 (커맨드라인 인자로 덮어쓸 수 있음) ---
DEFAULT_MODEL_NAMES = [
    "BAAI/bge-reranker-v2-m3",
    "dragonkue/bge-reranker-v2-m3-ko",
    "sigridjineth/ko-reranker-v1.1",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "jinaai/jina-reranker-v2-base-multilingual",
    "telepix/PIXIE-Spell-Reranker-Preview-0.6B",
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    "tomaarsen/Qwen3-Reranker-4B-seq-cls",
    "tomaarsen/Qwen3-Reranker-8B-seq-cls",
    "Dongjin-kr/ko-reranker",
    "upskyy/ko-reranker-8k",
    "mixedbread-ai/mxbai-rerank-large-v2",
    "BAAI/bge-reranker-v2-gemma",
]
DEFAULT_TASKS = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "XPQARetrieval",
    "MultiLongDocRetrieval",
    "MIRACLRetrieval",
    "MrTidyRetrieval"
]
DEFAULT_GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
V2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS_RESULTS_DIR = V2_ROOT / "eval/results/stage1/top_1k_qrels"
DEFAULT_OUTPUT_DIR = V2_ROOT / "eval/results/stage2"

assert V2_ROOT.exists(), f"V2_ROOT does not exist: {V2_ROOT}"
assert DEFAULT_PREVIOUS_RESULTS_DIR.exists(), f"DEFAULT_PREVIOUS_RESULTS_DIR does not exist: {DEFAULT_PREVIOUS_RESULTS_DIR}"
assert DEFAULT_OUTPUT_DIR.exists(), f"DEFAULT_OUTPUT_DIR does not exist: {DEFAULT_OUTPUT_DIR}"
# -----------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MTEB Reranker 벤치마크를 병렬로 실행합니다.")
    parser.add_argument(
        "--model_names", nargs="+", default=DEFAULT_MODEL_NAMES, help="평가할 리랭커 모델 이름 또는 경로 리스트"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=DEFAULT_TASKS, help="평가할 MTEB 태스크 리스트"
    )
    parser.add_argument(
        "--gpu_ids", nargs="+", type=int, default=DEFAULT_GPU_IDS, help="사용할 GPU ID 리스트"
    )
    parser.add_argument(
        "--previous_results_dir", type=str, default=str(DEFAULT_PREVIOUS_RESULTS_DIR), help="1단계(BM25) 결과가 저장된 디렉토리"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="2단계(리랭킹) 최종 결과를 저장할 디렉토리"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="리랭킹에 사용할 상위 K개 문서 수"
    )
    parser.add_argument(
        "--verbosity", type=int, default=0, help="MTEB 로그 상세 수준 (0: 진행률 표시줄만, 1: 점수 표시, 2: 상세 정보, 3: 디버그용)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="리랭킹 배치 사이즈"
    )
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn', force=True)
    
    previous_results_dir = Path(args.previous_results_dir)
    output_dir = Path(args.output_dir)

    job_queue = Queue()
    gpu_queue = Queue()

    total_jobs = 0
    for model_name in args.model_names:
        for task in args.tasks:
            job_queue.put((model_name, task))
            total_jobs += 1
    
    for gpu_id in args.gpu_ids:
        gpu_queue.put(gpu_id)

    processes = []
    num_workers = len(args.gpu_ids)
    print(f"Starting {num_workers} workers on GPUs: {args.gpu_ids}")
    print(f"Total jobs to process: {total_jobs}")

    for _ in range(num_workers):
        p = Process(target=worker, args=(job_queue, gpu_queue, previous_results_dir, output_dir, args.top_k, args.verbosity, args.batch_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("All evaluation tasks completed.")


if __name__ == "__main__":
    main()

