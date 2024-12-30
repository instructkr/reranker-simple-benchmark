from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "AutoRAG"

DATASET_CONFIGS: dict[str, dict[str, str]] = {
    "AutoRAG": {
        "qa_path": str(DATA_DIR / "qa_v4.parquet"),
        "corpus_path": str(DATA_DIR / "ocr_corpus_v3.parquet"),
    },
}
