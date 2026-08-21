import inspect
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseRerankerWrapper(ABC):
    """
    Abstract base class for reranker wrappers used in MTEB evaluation.
    All wrappers should implement the predict method to be compatible with MTEB.
    """

    @abstractmethod
    def predict(self, sentences: List[Tuple[str, str]], **kwargs) -> List[float]:
        """
        Compute relevance scores for query-document pairs.

        Args:
            sentences: List of (query, document) tuples
            **kwargs: Additional arguments like batch_size, show_progress_bar, etc.

        Returns:
            List of float scores, one per pair
        """
        pass


class Qwen3RerankerWrapper(BaseRerankerWrapper):
    def __init__(self, model_name: str, **kwargs):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, **kwargs)
        self.qwen3_instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    def _format_query(self, query: str) -> str:
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        return f"{prefix}<Instruct>: {self.qwen3_instruction}\n<Query>: {query}\n"

    def _format_document(self, document: str) -> str:
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return f"<Document>: {document}{suffix}"

    def predict(self, sentences: List[Tuple[str, str]], **kwargs) -> List[float]:
        formatted_sentences = []
        for query, document in sentences:
            if document is None:
                document = ""
            formatted_query = self._format_query(query)
            formatted_document = self._format_document(document)
            formatted_sentences.append([formatted_query, formatted_document])

        return self.model.predict(formatted_sentences, **kwargs)


class MxbaiRerankerWrapper(BaseRerankerWrapper):
    def __init__(self, model_name: str, **kwargs):
        from mxbai_rerank import MxbaiRerankV2

        self.model = MxbaiRerankV2(model_name, **kwargs)

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32, **kwargs) -> List[float]:
        # mteb 1.38 은 (q, d, instruction) 3-tuple, mteb 2.x 어댑터는 (q, d) 2-tuple → 둘 다 지원.
        # ★ MxbaiRerankV2.predict 는 batch_size 인자가 없어 전달된 쌍 전체를 한 번에 처리한다.
        #   task 전체 쌍(수십만)을 그대로 넘기면 OOM(수백 GB) → 여기서 청크로 나눠 호출한다.
        queries = [s[0] for s in sentences]
        documents = [s[1] if len(s) > 1 and s[1] is not None else "" for s in sentences]
        out: List[float] = []
        bs = max(1, int(batch_size))
        for i in range(0, len(queries), bs):
            sc = self.model.predict(queries[i:i + bs], documents[i:i + bs])
            out.extend(sc.tolist() if hasattr(sc, "tolist") else list(sc))
        return out


class BGEGemmaRerankerWrapper(BaseRerankerWrapper):
    def __init__(self, model_name: str, **kwargs):
        from FlagEmbedding import FlagLLMReranker

        # FlagLLMReranker.compute_score 기본 max_length 는 작음(문서 과도 절단) → 명시 통일.
        self.max_length = kwargs.pop("max_length", 8192)
        self.model = FlagLLMReranker(model_name, **kwargs)

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32, **kwargs) -> List[float]:
        scores = self.model.compute_score(
            sentences, batch_size=batch_size, max_length=self.max_length, **kwargs
        )
        return scores


class JinaRerankerV3Wrapper(BaseRerankerWrapper):
    def __init__(self, model_name: str, **kwargs):
        from transformers import AutoModel, AutoTokenizer

        device = kwargs.pop('device', 'cuda')
        # jina rerank API 는 max_length 인자가 없다. 초장문(MLDR 최대 ~13만자)을 native 131k 컨텍스트로
        # 처리하면 극도로 느리고 다른 모델(8192 절단)과 불공정하므로, 문서를 max_doc_tokens 로 절단한다.
        self.max_doc_tokens = kwargs.pop('max_doc_tokens', 8192)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs
        )
        self.model.to(device)
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            self.tokenizer = None

    def _truncate(self, doc: str) -> str:
        if not doc or self.tokenizer is None or self.max_doc_tokens is None:
            return doc
        ids = self.tokenizer(doc, add_special_tokens=False, truncation=True,
                             max_length=self.max_doc_tokens)["input_ids"]
        return self.tokenizer.decode(ids)

    def predict(self, sentences: List[Tuple[str, str]], **kwargs) -> List[float]:
        query_groups = {}
        for idx, item in enumerate(sentences):
            query = item[0]
            document = item[1] if len(item) > 1 and item[1] else ""
            document = self._truncate(document)

            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((idx, document))

        scores = [0.0] * len(sentences)

        # rerank 내부 문서 절단 길이를 max_doc_tokens 로 명시적으로 넘긴다.
        # v3: rerank 기본값이 max_doc_length=2048 이라, 넘기지 않으면 wrapper 가 8192 로 미리
        #     잘라도 내부에서 2048 로 재절단돼 기록된 _max_length=8192 와 불일치한다.
        # v3.5: max_doc_length 인자 자체가 없어 넘기면 TypeError 이므로, 시그니처 지원 시에만 전달.
        rerank_kwargs = {}
        if "max_doc_length" in inspect.signature(self.model.rerank).parameters:
            rerank_kwargs["max_doc_length"] = self.max_doc_tokens

        for query, doc_pairs in query_groups.items():
            if not doc_pairs:
                continue

            indices, docs = zip(*doc_pairs)

            results = self.model.rerank(
                query=query,
                documents=list(docs),
                **rerank_kwargs
            )

            for result in results:
                idx = indices[result['index']]
                scores[idx] = result['relevance_score']

        return scores


class QwenSeqClsWrapper(BaseRerankerWrapper):
    """tomaarsen/Qwen3-Reranker-*-seq-cls native.
    sentence-transformers 5.7 CrossEncoder 는 4B/8B 의 chat_template 을 pair 에 적용하려다 실패하므로
    (0.6B 는 chat_template 이 없어 통과), AutoModelForSequenceClassification 으로 직접 스코어링한다.
    공식 Qwen reranker 프롬프트 + 문서 truncation(assistant suffix 보존). score = seq-cls 로짓(num_labels=1)."""

    INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
    Q_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
               'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
               '<|im_end|>\n<|im_start|>user\n')
    D_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_name: str, **kwargs):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = kwargs.pop("device", "cuda")
        self.device = device
        self.max_length = kwargs.pop("max_length", 8192)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, **kwargs
        ).eval().to(device)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _build(self, query: str, doc: str) -> str:
        fq = f"{self.Q_PREFIX}<Instruct>: {self.INSTRUCT}\n<Query>: {query}\n"
        doc_prefix = "<Document>: "
        n_fixed = len(self.tokenizer(fq + doc_prefix + self.D_SUFFIX, add_special_tokens=True)["input_ids"])
        budget = max(16, self.max_length - n_fixed - 8)  # 문서만 예산에 맞게 잘라 suffix 보존
        d_ids = self.tokenizer(doc, add_special_tokens=False)["input_ids"][:budget]
        return fq + doc_prefix + self.tokenizer.decode(d_ids) + self.D_SUFFIX

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 16, **kwargs) -> List[float]:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                texts = [self._build(s[0], s[1] if len(s) > 1 and s[1] else "") for s in sentences[i:i + batch_size]]
                enc = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                score = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                out.extend(score.float().cpu().tolist())
        return out


class BGEGemmaNativeWrapper(BaseRerankerWrapper):
    """BAAI/bge-reranker-v2-gemma — transformers 5.x 에서 FlagEmbedding 의 `prepare_for_model` 이
    제거돼 동작 불가하므로, FlagEmbedding decoder-only reranker 의 포맷을 그대로 직접 구현한다.
    입력: [bos]+query 토큰 + '\\n' + passage 토큰(only_second 절단) + '\\n' + prompt 토큰.
    점수: 마지막 위치의 'Yes' 토큰 로짓 (FlagLLMReranker 와 동일)."""

    PROMPT = ("Given a query A and a passage B, determine whether the passage contains an answer "
              "to the query by providing a prediction of either 'Yes' or 'No'.")

    def __init__(self, model_name: str, **kwargs):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = kwargs.pop("device", "cuda")
        if isinstance(device, (list, tuple)):
            device = device[0]
        self.device = device
        self.max_length = kwargs.pop("max_length", 512)
        dtype = torch.bfloat16 if kwargs.pop("use_bf16", True) else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()
        self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        self.sep_ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
        self.prompt_ids = self.tokenizer(self.PROMPT, add_special_tokens=False)["input_ids"]
        self.bos = self.tokenizer.bos_token_id
        self.query_max_length = self.max_length * 3 // 4
        self.encode_max_length = self.max_length + len(self.sep_ids) + len(self.prompt_ids)

    def _build_ids(self, query: str, passage: str) -> List[int]:
        q_ids = self.tokenizer(query, add_special_tokens=False, truncation=True,
                               max_length=self.query_max_length)["input_ids"]
        p_ids = self.tokenizer(passage, add_special_tokens=False, truncation=True,
                               max_length=self.max_length)["input_ids"]
        head = ([self.bos] if self.bos is not None and self.bos != self.tokenizer.pad_token_id else []) + q_ids + self.sep_ids
        budget = self.encode_max_length - len(head)   # only_second: passage 만 절단
        p_ids = p_ids[:max(0, budget)]
        return head + p_ids + self.sep_ids + self.prompt_ids

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 16, **kwargs) -> List[float]:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                seqs = [self._build_ids(s[0], s[1] if len(s) > 1 and s[1] else "") for s in batch]
                maxlen = max(len(s) for s in seqs)
                pad = self.tokenizer.pad_token_id
                input_ids, attn = [], []
                for s in seqs:
                    padlen = maxlen - len(s)
                    input_ids.append([pad] * padlen + s)   # left padding
                    attn.append([0] * padlen + [1] * len(s))
                input_ids = torch.tensor(input_ids, device=self.device)
                attn = torch.tensor(attn, device=self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits[:, -1, self.yes_loc]
                out.extend(logits.float().cpu().tolist())
        return out


class NemotronRerankerWrapper(BaseRerankerWrapper):
    """nvidia/llama-nemotron-rerank-1b-v2 (LlamaBidirectionalForSequenceClassification, custom_code).
    ST 통합이 없어 CrossEncoder 로 못 씀 → README 레시피대로 AutoModelForSequenceClassification +
    프롬프트 'question:{q} \\n \\n passage:{p}' + left padding, seq-cls logit 을 점수로 반환."""

    def __init__(self, model_name: str, **kwargs):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = kwargs.pop('device', 'cuda')
        self.max_length = kwargs.pop('max_length', 8192)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, **kwargs
        ).eval()
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = self.model.to(device)

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32, **kwargs) -> List[float]:
        import torch

        texts = [
            f"question:{p[0]} \n \n passage:{(p[1] if len(p) > 1 and p[1] else '')}"
            for p in sentences
        ]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                batch = self.tokenizer(
                    chunk, padding=True, truncation=True,
                    return_tensors="pt", max_length=self.max_length,
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits.view(-1).float().cpu()
                out.extend(logits.tolist())
        return out