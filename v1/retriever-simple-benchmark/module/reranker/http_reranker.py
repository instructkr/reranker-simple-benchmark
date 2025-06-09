"""
HttpReranker: send rerank requests to a remote HTTP endpoint.
"""
import json
from typing import List, Tuple
from urllib import request as urllib_request, error as urllib_error

from .base import BaseReranker


class HttpReranker(BaseReranker):  # type: ignore[misc]
    """
    Reranker that queries a remote HTTP API endpoint for reranking.
    Expects the endpoint to accept POST with JSON {query, texts, truncate}
    and return JSON with a 'results' list of {index, score, text}.
    """
    def __init__(self, endpoint_url: str, truncate: bool = True, timeout: int = 60) -> None:
        self.endpoint_url = endpoint_url
        self.truncate = truncate
        self.timeout = timeout

    def compute_score_batch(
        self, query: str, docs: List[str], normalize: bool = False
    ) -> List[float]:
        # Build payload
        payload = {"query": query, "texts": docs, "truncate": self.truncate, 'instructions': '당신은 한국어 문서 검색 시스템의 리랭커입니다. 사용자의 질문에 가장 관련성이 높은 문서를 찾아 순위를 매겨야 합니다.   다음 기준으로 문서의 관련성을 평가하세요:  1. 질문의 핵심 키워드와 문서 내용의 일치도  2. 질문이 요구하는 정보의 구체적인 포함 여부  3. 문맥적 관련성과 의미적 유사도  특히 주의할 점:  - 질문의 언어(한국어/영어)와 관계없이 의미적으로 관련된 문서를 찾으세요  - 고유명사, 인명, 기관명 등은 다양한 표기가 가능함을 고려하세요  - 직접적인 답변이 없더라도 관련 정보가 포함된 문서도 중요합니다'}
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib_error.HTTPError as e:
            raise RuntimeError(f"HTTP error {e.code}: {e.reason}") from e
        except urllib_error.URLError as e:
            raise RuntimeError(f"URL error: {e.reason}") from e
        # Parse response
        resp_json = json.loads(body.decode("utf-8"))
        results = resp_json.get("results", [])
        # Prepare scores defaulting to 0.0
        scores: List[float] = [0.0] * len(docs)
        for item in results:
            idx = item.get("index")
            score = item.get("score")
            if isinstance(idx, int) and 0 <= idx < len(docs) and isinstance(score, (int, float)):
                scores[idx] = float(score)
        return scores

    def compute_score(
        self, pairs: List[Tuple[str, str]], normalize: bool = True
    ) -> List[float]:
        # Fallback: score each pair via batch of size 1
        scores: List[float] = []
        for q, doc in pairs:
            batch_scores = self.compute_score_batch(q, [doc], normalize=False)
            scores.append(batch_scores[0])
        if normalize:
            import numpy as np

            arr = np.array(scores, dtype=np.float32)
            min_s = float(arr.min())
            max_s = float(arr.max())
            if max_s - min_s > 1e-8:
                scores = [(s - min_s) / (max_s - min_s) for s in scores]
            else:
                scores = [0.0 for _ in scores]
        return scores