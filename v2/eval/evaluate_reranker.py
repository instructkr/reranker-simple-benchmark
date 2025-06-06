import os

import mteb
import torch

from mteb import MTEB
from sentence_transformers import CrossEncoder, SentenceTransformer

cross_encoder_model_names = [
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

tasks = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "MIRACLRetrieval",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "XPQARetrieval",
]

batch_size = 2048

for cross_encoder_model_name in cross_encoder_model_names:
    print(f"Evaluating model: {cross_encoder_model_name}")
    
    cross_encoder = CrossEncoder(cross_encoder_model_name, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16})
    
    output_dir = os.path.join("./results/stage2", cross_encoder_model_name)
    
    for task in tasks:
        print(f"Running tasks: {task} / {cross_encoder_model_name}")

        tasks_mteb = mteb.get_tasks(tasks=[task], languages=["kor-Kore", "kor-Hang", "kor_Hang"])
        eval_splits = ["test"]
        evaluation = MTEB(tasks=tasks_mteb)

        if os.path.exists(os.path.join(previous_results_dir, task + ".json")):
            print(f"Previous results found: {task}")
            previous_results = os.path.join(previous_results_dir, task + ".json")

            evaluation.run(
                cross_encoder,
                top_k=100,
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
    
    print(f"Completed evaluation for model: {cross_encoder_model_name}")