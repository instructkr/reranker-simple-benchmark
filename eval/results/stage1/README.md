# Stage 1 Retrieval (BM25)

Stage 2 reranking 의 1차 후보(query 당 BM25 top-1000, 정답 문서 포함)를 만드는 단계입니다. 명령은 repo root 기준입니다.

최적의 성능을 보여주는 한국어 tokenizer를 선정하기 위해 tokenizer별 평가를 진행하였습니다. 
## Evaluation Code
```bash
# BM25 Stage 1 검색 (query 당 top-1000, 정답 문서 포함). 토크나이저는 Mecab/Kiwi/Okt/Kkma 중 선택.
uv run python eval/retrieve_stage1_bm25.py --tokenizer Mecab --datasets all

# 공식 MTEB(kor, v2) 기준 task 마이닝 (mteb 2.x, 표준 eval_splits, 예: MLDR=dev+test):
uv run python eval/retrieve_stage1_bm25.py --tokenizer Mecab --kor_tasks MultiLongDocRetrieval LawIRKo
```
## Leaderboard
```bash
cd eval
uv run streamlit run leaderboard_bm25.py
```
## Results
| Model | Average Recall@10 | Average Precision@10 | Average NDCG@10 | Average F1@10 |
|-------|----------------|-------------------|--------------|------------|
| Mecab | **0.8731**     | 0.1000            | 0.7433       | **0.1783** |
| Okt   | 0.8655         | **0.1001**        | **0.7474**   | 0.1783     |
| Kkma  | 0.8504         | 0.0982            | 0.7358       | 0.1749     |
| Kiwi  | 0.8443         | 0.0961            | 0.7210       | 0.1715     |

top-k 10에서 가장 높은 성능을 보인 **Mecab** tokenizer를 사용하여, Stage 1 Retrieval을 진행하였습니다.
