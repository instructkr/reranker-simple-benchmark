import typer
import os
import json
import datetime

from module.reranker.base import BaseReranker
from module.reranker import determine_reranker_class
from module.reranker.http_reranker import HttpReranker
from data import DATASET_CONFIGS
from utils.data import load_data
from utils.evaluate import evaluate_model

app = typer.Typer(help="CLI for model evaluation using Typer.")


@app.command()
def evaluate(
    # 1) Required option for 'type' (only 'cross-encoder' supported)
    model_type: str = typer.Option(
        ...,
        "--type",
        help="Which model approach to use. Currently only 'cross-encoder' is supported.",
    ),
    # 2) Required option for model name/path
    model_name: str = typer.Option(
        ...,
        "--model_name",
        help="Model name or path (e.g. 'sigridjineth/ko-reranker-v1.1').",
    ),
    # 3) Reranker class type (huggingface, flagreranker, etc.)
    model_class: str = typer.Option(
        "huggingface",
        "--model_class",
        help="Which model class to use. Possible: 'huggingface', 'flagreranker', 'flagllmreranker', 'flaglayerwise', 'mxbai'. Default: huggingface.",
    ),
    # 4) Data type name (only supports 'AutoRAG')
    datatype_name: str = typer.Option(
        "AutoRAG",
        "--datatype_name",
        help="Which dataset to use. Currently only 'AutoRAG' is supported.",
    ),
    # 5) Additional option: use_fp16
    use_fp16: bool = typer.Option(
        True,
        "--use-fp16",
        help="Use FP16 on GPU if available.",
    ),
    # 6) Only for HTTP reranker: truncate texts before sending
    truncate: bool = typer.Option(
        True,
        "--truncate/--no-truncate",
        help="Whether to truncate texts when querying remote HTTP endpoint; only used for model_class 'http' or 'api'.",
    ),
):
    """
    Evaluate a Cross-Encoder model on the given dataset.

    --type cross-encoder
    --model_name: The model path or identifier
    --model_class: huggingface or flagreranker
    --datatype_name: Only 'AutoRAG' is supported at this time
    --use_fp16: Whether to use FP16 on GPU

    Example usage:
      uv run ./retriever-simple-benchmark/main.py evaluate \
        --type cross-encoder \
        --model_name "sigridjineth/ko-reranker-v1.1" \
        --model_class huggingface \
        --datatype_name AutoRAG
        
      # Example using MxbaiReranker:
      uv run ./retriever-simple-benchmark/main.py evaluate \
        --type cross-encoder \
        --model_name "mixedbread-ai/mxbai-rerank-base-v2" \
        --model_class mxbai \
        --datatype_name AutoRAG
    """

    # 1) Validate model_type
    if model_type.lower() != "cross-encoder":
        typer.echo(
            f"We only support 'cross-encoder' at this time. Received: {model_type}"
        )
        raise typer.Exit(code=1)

    # 2) Check if requested data type is 'AutoRAG'
    if datatype_name not in DATASET_CONFIGS:
        typer.echo(
            f"datatype_name='{datatype_name}' is not supported. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
        raise typer.Exit(code=1)

    # Retrieve dataset paths from DATASET_CONFIGS
    dataset_info = DATASET_CONFIGS[datatype_name]
    qa_path = dataset_info["qa_path"]
    corpus_path = dataset_info["corpus_path"]

    typer.echo(f"Using data type '{datatype_name}'...")
    typer.echo(f"  QA Path: {qa_path}")
    typer.echo(f"  Corpus Path: {corpus_path}")

    qd_df, corpus_df, valid_dict = load_data(qa_path, corpus_path)

    # 3) Instantiate the Reranker
    typer.echo(f"Model Type: {model_type}")
    typer.echo(f"Model Name: {model_name}")
    typer.echo(f"Model Class: {model_class}")
    typer.echo(f"FP16 enabled: {use_fp16}")
    typer.echo(f"Truncate texts: {truncate}")
    # Instantiate the appropriate reranker
    model_class_lower = model_class.lower()
    if model_class_lower in ("http", "api"):
        typer.echo(f"Using HTTP Reranker endpoint: {model_name}")
        reranker: BaseReranker = HttpReranker(endpoint_url=model_name, truncate=truncate)
    else:
        reranker: BaseReranker = determine_reranker_class(model_class, model_name, use_fp16)

    # 4) Evaluate the Model
    typer.echo("Starting evaluation...")

    # Modified evaluate_model to also return timing info
    accuracies, f1_scores, recalls, precisions, total_time, avg_time = evaluate_model(
        corpus_df, qd_df, valid_dict, reranker
    )

    # 5) Print & Save Results
    typer.echo("\nEvaluation Results (k=1,3,5,10):")
    k_values = [1, 3, 5, 10]
    for k in k_values:
        typer.echo(f"  Accuracy@{k}:  {accuracies[k]:.4f}")
        typer.echo(f"  F1@{k}:        {f1_scores[k]:.4f}")
        typer.echo(f"  Recall@{k}:    {recalls[k]:.4f}")
        typer.echo(f"  Precision@{k}: {precisions[k]:.4f}")

    typer.echo(f"\nTotal inference time (all queries): {total_time:.2f} sec")
    typer.echo(f"Average inference time (per query): {avg_time:.4f} sec")

    typer.echo("Evaluation complete.")

    os.makedirs("result", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"result/{timestamp}.json"

    results_dict: dict = {
        "timestamp": timestamp,
        "datatype_name": datatype_name,
        "qa_path": qa_path,
        "corpus_path": corpus_path,
        "model_type": model_type,
        "model_name": model_name,
        "model_class": model_class,
        "use_fp16": use_fp16,
        "metrics": {
            "accuracy": {str(k): accuracies[k] for k in k_values},
            "f1": {str(k): f1_scores[k] for k in k_values},
            "recall": {str(k): recalls[k] for k in k_values},
            "precision": {str(k): precisions[k] for k in k_values},
        },
        "inference_times": {
            "total_inference_time_sec": total_time,
            "avg_inference_time_per_query_sec": avg_time,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    typer.echo(f"Saved results to: {output_path}")


if __name__ == "__main__":
    app()
