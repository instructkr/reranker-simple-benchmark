from __future__ import annotations

from bm25s.tokenization import Tokenizer, Tokenized

import logging
from functools import partial

import mteb
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

from konlpy.tag import Kkma
from kiwipiepy import Kiwi
from konlpy.tag import Okt

import re

# from sentence_transformers import SentenceTransformer
import argparse

logger = logging.getLogger(__name__)


def clean_text(text):
    # 허용 문자: 한글, 숫자, 영문, 공백, 특수문자 (기본적인 기호들)
    pattern = r"[^가-힣a-zA-Z0-9\s~!@#$%^&*()_\-+=\[\]{}|\\;:'\",.<>/?`]"
    cleaned = re.sub(pattern, "-", text)
    return cleaned


class Kiwi_(Kiwi):
    def morphs(kiwi, text):
        return [morph.form for morph in kiwi.tokenize(text)]


class Kokenizer(Tokenizer):
    def kokenize(self, texts, tokenizer):
        # 2. 문장별 형태소 분석 (어간 추출 포함)
        corpus_tokens = []
        for sentence in texts:
            try:
                corpus_tokens.append(tokenizer.morphs(sentence))
            except UnicodeDecodeError as e:
                print(f"[{sentence}] ❗인코딩 에러 발생: {e}")
                corpus_tokens.append(tokenizer.morphs(clean_text(sentence)))

        # 3. Vocab 만들기 ("" 포함해서 index 0)
        vocab = {"": 0}
        index = 1
        for tokens in corpus_tokens:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = index
                    index += 1

        # 4. 각 문장을 vocab index로 변환
        indexed_tokens = [
            [vocab[token] for token in tokens] for tokens in corpus_tokens
        ]

        return Tokenized(ids=indexed_tokens, vocab=vocab)


def bm25_loader(**kwargs):
    model_name = kwargs.get("model_name", "BM25")
    task_name = kwargs.get("task_name", "bm25s")
    tokenizer_name = kwargs.get("tokenizer_name", None)
    top_k = kwargs.get("top_k", 1000)
    requires_package(bm25_loader, "bm25s", model_name, "pip install mteb[bm25s]")
    import bm25s
    import Stemmer

    class BM25Search(DRESModel, Wrapper):
        """BM25 search"""

        def __init__(
            self,
            previous_results: str = None,
            stopwords: str = "en",
            stemmer_language: str | None = "english",
            **kwargs,
        ):
            super().__init__(
                model=None,
                batch_size=1,
                corpus_chunk_size=1,
                previous_results=previous_results,
                **kwargs,
            )

            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )
            self.task_name = task_name
            self.tokenizer_name = tokenizer_name
            self.top_k = top_k

        @classmethod
        def name(self):
            return "bm25s"

        def search(
            self,
            corpus: dict[str, dict[str, str]],
            queries: dict[str, str | list[str]],
            top_k: int,
            score_function: str,
            return_sorted: bool = False,
            **kwargs,
        ) -> dict[str, dict[str, float]]:
            logger.info("Encoding Corpus...")
            corpus_ids = list(corpus.keys())
            corpus_with_ids = [
                {
                    "doc_id": cid,
                    **(
                        {"text": corpus[cid]}
                        if isinstance(corpus[cid], str)
                        else corpus[cid]
                    ),
                }
                for cid in corpus_ids
            ]

            corpus_texts = [
                "\n".join([doc.get("title", ""), doc["text"]])
                for doc in corpus_with_ids
            ]  # concatenate all document values (title, text, ...)
            encoded_corpus = self.encode(
                corpus_texts,
                task_name=self.task_name,
            )

            logger.info(
                f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
            )

            # Create the BM25 model and index the corpus
            retriever = bm25s.BM25()
            retriever.index(encoded_corpus)

            logger.info("Encoding Queries...")
            query_ids = list(queries.keys())
            self.results = {qid: {} for qid in query_ids}
            queries_texts = [queries[qid] for qid in queries]

            query_token_strs = self.encode(queries_texts, return_ids=False)

            logger.info(f"Retrieving Results... {len(queries):,} queries")

            queries_results, queries_scores = retriever.retrieve(
                query_token_strs, corpus=corpus_with_ids, k=self.top_k
            )

            # Iterate over queries
            for qi, qid in enumerate(query_ids):
                doc_id_to_score = {}
                query_results = queries_results[qi]
                scores = queries_scores[qi]
                doc_id_to_score = {}

                # Iterate over results
                for ri in range(len(query_results)):
                    doc = query_results[ri]
                    score = scores[ri]
                    doc_id = doc["doc_id"]

                    doc_id_to_score[doc_id] = float(score)

                self.results[qid] = doc_id_to_score

            return self.results

        def encode(self, texts: list[str], task_name=None, **kwargs):
            """Encode input text as term vectors"""

            # 1. 토크나이저 오브젝트 생성
            if self.tokenizer_name == "Okt":
                okt = Okt()
                koken = Kokenizer(stopwords=None)
                print("Okt로 Tokenizing하는 중입니다.")
                return koken.kokenize(texts, tokenizer=okt)
            elif self.tokenizer_name == "Kkma":
                kkma = Kkma()
                koken = Kokenizer(stopwords=None)
                print("Kkma로 Tokenizing하는 중입니다.")
                return koken.kokenize(texts, tokenizer=kkma)
            elif self.tokenizer_name == "Kiwi":
                kiwi = Kiwi_()
                koken = Kokenizer(stopwords=None)
                print("kiwi로 Tokenizing하는 중입니다.")
                return koken.kokenize(texts, tokenizer=kiwi)

    return BM25Search(**kwargs)


def retrieve_bm25_with_tokenize(
    tokenizer_name_list: list = "all", data_list: list = "all"
):
    tokenizer_name_list = (
        ["Okt", "Kkma", "Kiwi"] if tokenizer_name_list == "all" else tokenizer_name_list
    )
    data_list = (
        ["PublicHealthQA", "Ko-StrategyQA", "AutoRAGRetrieval", "BelebeleRetrieval"]
        if data_list == "all"
        else data_list
    )
    # data_list = ["PublicHealthQA", "MultiLongDocRetrieval", "Ko-StrategyQA", "AutoRAGRetrieval", "BelebeleRetrieval", "MIRACLRetrieval", "MrTidyRetrieval" ]

    print(
        f"다음의 토크나이저 {tokenizer_name_list}와, 데이터셋 {data_list}에 대해 MTEB Retrival Evaluation을 실행합니다."
    )

    for tokenizer_name in tokenizer_name_list:
        print("#" * 30, f"{tokenizer_name}을 활용합니다", "#" * 30)
        for data in data_list:
            print(data, "start")
            tasks = mteb.get_tasks(tasks=[data], languages=["kor"])

            if data in ["PublicHealthQA", "AutoRAGRetrieval", "BelebeleRetrieval"]:
                if hasattr(tasks[0], "load_data"):
                    tasks[0].load_data()

                if tasks[0].corpus.get("korean", None):
                    len_data = len(
                        tasks[0].corpus.get("korean", None).get("test", None)
                    )
                elif tasks[0].corpus.get("kor_Hang-kor_Hang", None):
                    len_data = len(
                        tasks[0].corpus.get("kor_Hang-kor_Hang", None).get("test", None)
                    )
                elif tasks[0].corpus.get("test", None):
                    len_data = len(tasks[0].corpus.get("test", None))
                elif tasks[0].corpus.get("dev", None):
                    len_data = len(tasks[0].corpus.get("dev", None))
                else:
                    print("데이터의 길이를 알 수 없습니다.")

                print(data, f"데이터의 길이는 {len_data}입니다")
            else:
                len_data = 1000
                print(data, f"데이터의 길이는 {len_data} 이상입니다")

            bm25_s = ModelMeta(
                loader=partial(
                    bm25_loader,
                    model_name="bm25s",
                    task_name="bm25s",
                    tokenizer_name=tokenizer_name,
                    top_k=len_data if len_data < 1000 else 1000,
                ),  # type: ignore
                name="bm25s",
                languages=["kor_Hang"],
                open_weights=True,
                revision="0_1_10",
                release_date="2024-07-10",  ## release of version 0.1.10
                n_parameters=None,
                memory_usage_mb=None,
                embed_dim=None,
                license=None,
                max_tokens=None,
                reference="https://github.com/xhluca/bm25s",
                similarity_fn_name=None,
                framework=[],
                use_instructions=False,
                public_training_code="https://github.com/xhluca/bm25s",
                public_training_data=None,
                training_datasets=None,
            ).load_model()

            evaluation = mteb.MTEB(tasks=tasks)
            results = evaluation.run(
                bm25_s,
                output_folder=f"results/stage1/bm25_{tokenizer_name}",
                save_predictions=True,
            )
            print(
                "#" * 30,
                f"[{data}] evaluation completed @ [{tokenizer_name}]",
                "#" * 30,
            )
            print(results[0].scores)
            print()


parser = argparse.ArgumentParser()


def main():
    parser = argparse.ArgumentParser(
        description="리랭킹 이전에 BM25로 1000개의 검색 결과를 뽑는 함수입니다. 한국어 토크나이저를 입력할 수 있습니다."
    )
    parser.add_argument(
        "--tokenizer_list", nargs="+", default="all", type=str, help="ex) Okt Kkma Kiwi"
    )
    parser.add_argument(
        "--data_list",
        nargs="+",
        default="all",
        type=str,
        help="ex) PublicHealthQA , Ko-StrategyQA, AutoRAGRetrieval, BelebeleRetrieval 등",
    )

    args = parser.parse_args()

    retrieve_bm25_with_tokenize(args.tokenizer_list, args.data_list)


if __name__ == "__main__":
    main()
