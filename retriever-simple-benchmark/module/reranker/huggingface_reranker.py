import torch
import numpy as np
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HuggingFaceReranker:
    def __init__(self, model_path, use_fp16=True, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        if use_fp16 and self.device.type == "cuda":
            model.half()

        if torch.cuda.device_count() > 1:
            model = DataParallel(model)

        self.model = model.to(self.device)
        self.model.eval()

        self.max_length = max_length

    @torch.no_grad()
    def compute_score_batch(
        self, query: str, doc_texts: list[str], normalize=False
    ) -> list[float]:
        """
        Compute scores for a batch of docs at once.
        doc_texts: e.g.  [doc1, doc2, ...]
        Return: list of float scores
        """

        # Tokenize in a single batch
        inputs = self.tokenizer(
            [query] * len(doc_texts),  # replicate the query for each doc
            doc_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        # For a cross-encoder: model output is [batch_size, 1] or [batch_size, 2]
        # Suppose it's a single regression head => shape [batch_size, 1]
        logits = outputs.logits.squeeze(dim=-1).float()  # shape: [batch_size]

        scores = logits.cpu().numpy().tolist()
        return scores

    def compute_score(self, pairs, normalize=True) -> list[float]:
        """
        pairs: [(query, doc_text), ...]
        This is a simpler method, might not be multi-GPU batched.
        """

        # For backward-compat.
        scores = []
        for q, doc in pairs:
            batch_scores = self.compute_score_batch(q, [doc], normalize=False)
            scores.append(batch_scores[0])

        if normalize:
            min_s = float(np.min(scores))
            max_s = float(np.max(scores))
            if (max_s - min_s) > 1e-8:
                scores = [(s - min_s) / (max_s - min_s) for s in scores]
            else:
                scores = [0.0 for _ in scores]
        return scores
