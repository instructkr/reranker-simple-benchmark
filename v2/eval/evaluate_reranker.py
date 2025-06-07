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

from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
_original_load_results_file = DenseRetrievalExactSearch.load_results_file

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

            tasks_mteb = mteb.get_tasks(tasks=[task], languages=["kor-Kore", "kor-Hang", "kor_Hang"])
            evaluation = MTEB(tasks=tasks_mteb)

            if os.path.exists(os.path.join(previous_results_dir, task + "_id.jsonl")):
                print(f"Previous results found: {task}")
                previous_results = os.path.join(previous_results_dir, task + "_id.jsonl")

                evaluation.run(
                    cross_encoder,
                    top_k=1000,
                    save_predictions=False,
                    output_folder=output_dir,
                    previous_results=previous_results,
                    batch_size=batch_size
                )
            else:
                print(f"Previous results not found: {task}")
                evaluation.run(
                    cross_encoder,
                    save_predictions=False,
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