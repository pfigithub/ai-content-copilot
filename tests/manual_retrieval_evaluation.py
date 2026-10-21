import asyncio
import json
from typing import Any

from langchain_core.documents import Document

from app.retrieval.document_splitter import split_document
from app.retrieval.qdrant_store import QdrantStore


# Keep evaluation data isolated from the production collection.
COLLECTION_NAME = "retrieval_evaluation"

# Evaluate several thresholds before choosing a candidate.
BASE_THRESHOLDS = (
    0.25,
    0.28,
    0.30,
    0.35,
    0.39,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)

RECALL_AT_K_VALUES = (1, 3, 5)
MAX_RETRIEVAL_K = max(RECALL_AT_K_VALUES)


# Each source contains one short document for source-level evaluation.
DATASET = [
    {
        "source": "ai.txt",
        "text": (
            "هوش مصنوعی می‌تواند فرایند تحقیق و "
            "تولید محتوا را خودکار کند و به "
            "تولیدکنندگان محتوا در ایده‌پردازی، "
            "تحلیل و افزایش سرعت کار کمک کند."
        ),
    },
    {
        "source": "agriculture.txt",
        "text": (
            "آبیاری قطره‌ای و استفاده از روش‌های "
            "نوین آبیاری می‌تواند مصرف آب در "
            "کشاورزی را کاهش دهد و بهره‌وری "
            "منابع آب را افزایش دهد."
        ),
    },
    {
        "source": "network.txt",
        "text": (
            "شبکه‌های کامپیوتری برای انتقال داده "
            "میان دستگاه‌ها استفاده می‌شوند و "
            "پروتکل‌های ارتباطی نقش مهمی در "
            "مدیریت این ارتباط دارند."
        ),
    },
    {
        "source": "economics.txt",
        "text": (
            "افزایش نرخ بهره می‌تواند هزینه "
            "استقراض را افزایش دهد و بر "
            "سرمایه‌گذاری، مصرف و فعالیت اقتصادی "
            "تأثیر بگذارد."
        ),
    },
    {
        "source": "education.txt",
        "text": (
            "برنامه‌ریزی منظم مطالعه و تقسیم "
            "هدف‌های آموزشی به فعالیت‌های کوچک‌تر "
            "می‌تواند به دانش‌آموزان در ایجاد "
            "عادت‌های پایدار یادگیری کمک کند."
        ),
    },
    {
        "source": "sports.txt",
        "text": (
            "تمرین منظم و آمادگی جسمانی می‌تواند "
            "عملکرد ورزشکاران را بهبود دهد و "
            "خطر برخی آسیب‌های ورزشی را کاهش دهد."
        ),
    },
]


# Paraphrases test retrieval beyond close wording matches.
EVALUATION_CASES = [
    {
        "query": "چطور می‌توان مصرف آب کشاورزی را کاهش داد؟",
        "expected_source": "agriculture.txt",
    },
    {
        "query": (
            "چه روشی باعث می‌شود آب کمتری برای "
            "آبیاری زمین‌های کشاورزی مصرف شود؟"
        ),
        "expected_source": "agriculture.txt",
    },
    {
        "query": (
            "برای افزایش بهره‌وری منابع آب در "
            "مزرعه چه راهکاری وجود دارد؟"
        ),
        "expected_source": "agriculture.txt",
    },
    {
        "query": "هوش مصنوعی چگونه به تولید محتوا کمک می‌کند؟",
        "expected_source": "ai.txt",
    },
    {
        "query": (
            "برای سریع‌تر شدن ایده‌پردازی و آماده‌سازی "
            "محتوا از چه فناوری‌ای می‌توان کمک گرفت؟"
        ),
        "expected_source": "ai.txt",
    },
    {
        "query": (
            "کاربردهای هوش مصنوعی در تحقیق و "
            "تولید مطالب چیست؟"
        ),
        "expected_source": "ai.txt",
    },
    {
        "query": (
            "چه عواملی روی سرمایه‌گذاری و "
            "فعالیت اقتصادی تأثیر می‌گذارند؟"
        ),
        "expected_source": "economics.txt",
    },
    {
        "query": (
            "بالا رفتن نرخ بهره چه اثری بر "
            "گرفتن وام و سرمایه‌گذاری می‌گذارد؟"
        ),
        "expected_source": "economics.txt",
    },
    {
        "query": (
            "تغییر نرخ بهره چگونه می‌تواند اقتصاد، "
            "مصرف و سرمایه‌گذاری را تحت تأثیر قرار دهد؟"
        ),
        "expected_source": "economics.txt",
    },
    {
        "query": "چگونه می‌توان عادت مطالعه بهتری ایجاد کرد؟",
        "expected_source": "education.txt",
    },
    {
        "query": (
            "برای منظم کردن درس خواندن و رسیدن "
            "به هدف‌های آموزشی چه روشی مفید است؟"
        ),
        "expected_source": "education.txt",
    },
    {
        "query": (
            "چگونه تقسیم برنامه آموزشی به کارهای "
            "کوچک می‌تواند به یادگیری مداوم کمک کند؟"
        ),
        "expected_source": "education.txt",
    },
    {
        "query": "پروتکل‌ها در شبکه‌های کامپیوتری چه نقشی دارند؟",
        "expected_source": "network.txt",
    },
    {
        "query": (
            "دستگاه‌های کامپیوتری از چه طریقی "
            "اطلاعات را بین خود جابه‌جا می‌کنند؟"
        ),
        "expected_source": "network.txt",
    },
    {
        "query": (
            "اهمیت پروتکل‌های ارتباطی در "
            "تبادل داده چیست؟"
        ),
        "expected_source": "network.txt",
    },
    {
        "query": "چطور می‌توان عملکرد ورزشکاران را بهبود داد؟",
        "expected_source": "sports.txt",
    },
    {
        "query": (
            "چه چیزهایی به بهتر شدن نتیجه ورزشکاران "
            "و کم شدن آسیب‌ها کمک می‌کند؟"
        ),
        "expected_source": "sports.txt",
    },
    {
        "query": (
            "چرا آمادگی بدنی و تمرین مداوم "
            "برای عملکرد ورزشی اهمیت دارند؟"
        ),
        "expected_source": "sports.txt",
    },
]


# These queries have no directly relevant source in this dataset.
OUT_OF_DOMAIN_CASES = [
    "ترکیب شیمیایی آب از چه عناصری تشکیل شده است؟",
    "امپراتوری روم باستان در چه دوره‌ای شکل گرفت؟",
    "چگونه می‌توان یک شیرینی شکلاتی در خانه پخت؟",
    "سیاره زحل چه ویژگی‌هایی در حلقه‌های خود دارد؟",
    "چه تفاوتی میان نقاشی رنگ روغن و آبرنگ وجود دارد؟",
    "چرخه زندگی پروانه شامل چه مراحلی است؟",
]


def build_documents() -> list[Document]:
    """Split the evaluation dataset into documents for vector retrieval."""

    documents: list[Document] = []

    for item in DATASET:
        documents.extend(
            split_document(
                text=item["text"],
                source=item["source"],
                source_type="evaluation",
                chunk_size=500,
                chunk_overlap=50,
            )
        )

    return documents


def serialize_results(
    results: list[tuple[Document, float]],
) -> list[dict[str, Any]]:
    """Convert retrieval results into JSON-serializable records."""

    return [
        {
            "rank": rank,
            "source": document.metadata.get("source"),
            "chunk_index": document.metadata.get("chunk_index"),
            "score": round(float(score), 4),
        }
        for rank, (document, score) in enumerate(results, start=1)
    ]


def find_relevant_rank(
    results: list[tuple[Document, float]],
    expected_source: str,
) -> int | None:
    """Return the one-based rank of the expected source, if retrieved."""

    for rank, (document, _) in enumerate(results, start=1):
        if document.metadata.get("source") == expected_source:
            return rank

    return None


def summarize_scores(
    scores: list[float | None],
) -> dict[str, float | int] | None:
    """Summarize available scores without treating them as probabilities."""

    available_scores = [
        float(score)
        for score in scores
        if score is not None
    ]

    if not available_scores:
        return None

    return {
        "count": len(available_scores),
        "minimum": round(min(available_scores), 4),
        "maximum": round(max(available_scores), 4),
        "mean": round(
            sum(available_scores) / len(available_scores),
            4,
        ),
    }


def evaluate_threshold(
    threshold: float,
    positive_scores: list[float | None],
    negative_scores: list[float | None],
) -> dict[str, float | int]:
    """Measure answerability classification at one score threshold."""

    # A query is accepted only when its nearest result meets the threshold.
    true_positives = sum(
        score is not None and score >= threshold
        for score in positive_scores
    )
    false_negatives = len(positive_scores) - true_positives

    false_positives = sum(
        score is not None and score >= threshold
        for score in negative_scores
    )
    true_negatives = len(negative_scores) - false_positives

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    specificity_denominator = true_negatives + false_positives

    precision = (
        true_positives / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positives / recall_denominator
        if recall_denominator
        else 0.0
    )
    specificity = (
        true_negatives / specificity_denominator
        if specificity_denominator
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2

    return {
        "threshold": round(threshold, 4),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1_score, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
    }


def build_threshold_candidates(
    positive_scores: list[float | None],
    negative_scores: list[float | None],
) -> list[float]:
    """Combine fixed test thresholds with a data-derived midpoint candidate."""

    candidates = list(BASE_THRESHOLDS)

    available_positive = [
        score for score in positive_scores if score is not None
    ]
    available_negative = [
        score for score in negative_scores if score is not None
    ]

    # Add a midpoint only when the observed groups are separated.
    if available_positive and available_negative:
        minimum_positive = min(available_positive)
        maximum_negative = max(available_negative)

        if maximum_negative < minimum_positive:
            midpoint = (maximum_negative + minimum_positive) / 2
            candidates.append(round(midpoint, 4))

    return sorted(set(candidates))


async def main() -> None:
    """Rebuild the isolated evaluation collection and report retrieval metrics."""

    # Prevent accidental deletion if the collection name is changed.
    if COLLECTION_NAME != "retrieval_evaluation":
        raise RuntimeError(
            "Refusing to delete a collection other than "
            "'retrieval_evaluation'."
        )

    store = QdrantStore(collection_name=COLLECTION_NAME)

    # This deletes only the dedicated evaluation collection.
    if store.client.collection_exists(
        collection_name=COLLECTION_NAME,
    ):
        store.client.delete_collection(
            collection_name=COLLECTION_NAME,
        )

    documents = build_documents()
    inserted_ids = store.add_documents(documents)

    print("EVALUATION DATASET")
    print("documents:", len(documents))
    print("inserted:", len(inserted_ids))
    print("positive_queries:", len(EVALUATION_CASES))
    print("out_of_domain_queries:", len(OUT_OF_DOMAIN_CASES))
    print("retrieval_k:", MAX_RETRIEVAL_K)

    hit_counts = {
        cutoff: 0
        for cutoff in RECALL_AT_K_VALUES
    }
    reciprocal_rank_sum = 0.0
    positive_top_scores: list[float | None] = []

    # Run positive cases without a threshold to collect raw scores.
    for case_number, case in enumerate(EVALUATION_CASES, start=1):
        results = await store.similarity_search_with_score(
            query=case["query"],
            k=MAX_RETRIEVAL_K,
        )

        expected_source = case["expected_source"]
        relevant_rank = find_relevant_rank(
            results=results,
            expected_source=expected_source,
        )

        top_source = (
            results[0][0].metadata.get("source")
            if results
            else None
        )
        top_score = float(results[0][1]) if results else None
        top_1_match = top_source == expected_source

        positive_top_scores.append(top_score)

        for cutoff in RECALL_AT_K_VALUES:
            if relevant_rank is not None and relevant_rank <= cutoff:
                hit_counts[cutoff] += 1

        if relevant_rank is not None:
            reciprocal_rank_sum += 1.0 / relevant_rank

        print(f"\nPOSITIVE QUERY {case_number}")
        print("query:", case["query"])
        print("expected_source:", expected_source)
        print("top_source:", top_source)
        print("top_score:", round(top_score, 4) if top_score is not None else None)
        print("relevant_rank:", relevant_rank)
        print("top_1_match:", top_1_match)
        print(
            "ranked_results:",
            json.dumps(
                serialize_results(results),
                ensure_ascii=False,
                indent=2,
            ),
        )

    total_positive = len(EVALUATION_CASES)

    print("\nPOSITIVE EVALUATION SUMMARY")
    print("total_queries:", total_positive)
    print("top_1_hits:", hit_counts[1])
    print(
        "top_1_accuracy:",
        f"{hit_counts[1] / total_positive:.2%}",
    )

    for cutoff in RECALL_AT_K_VALUES:
        recall = hit_counts[cutoff] / total_positive
        print(f"recall_at_{cutoff}:", f"{recall:.2%}")

    print(
        f"mrr_at_{MAX_RETRIEVAL_K}:",
        f"{reciprocal_rank_sum / total_positive:.4f}",
    )

    negative_top_scores: list[float | None] = []

    # Run out-of-domain queries to inspect their nearest-score distribution.
    print("\nOUT-OF-DOMAIN EVALUATION")

    for case_number, query in enumerate(OUT_OF_DOMAIN_CASES, start=1):
        results = await store.similarity_search_with_score(
            query=query,
            k=MAX_RETRIEVAL_K,
        )

        nearest_source = (
            results[0][0].metadata.get("source")
            if results
            else None
        )
        nearest_score = (
            float(results[0][1])
            if results
            else None
        )
        negative_top_scores.append(nearest_score)

        print(f"\nOUT-OF-DOMAIN QUERY {case_number}")
        print("query:", query)
        print("nearest_source:", nearest_source)
        print(
            "nearest_score:",
            round(nearest_score, 4)
            if nearest_score is not None
            else None,
        )
        print(
            "ranked_results:",
            json.dumps(
                serialize_results(results),
                ensure_ascii=False,
                indent=2,
            ),
        )

    print("\nRAW TOP-SCORE SUMMARY")
    print(
        "positive_queries:",
        json.dumps(
            summarize_scores(positive_top_scores),
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(
        "out_of_domain_queries:",
        json.dumps(
            summarize_scores(negative_top_scores),
            ensure_ascii=False,
            indent=2,
        ),
    )

    # Compare thresholds on the cached top scores without another database pass.
    thresholds = build_threshold_candidates(
        positive_scores=positive_top_scores,
        negative_scores=negative_top_scores,
    )

    threshold_results = [
        evaluate_threshold(
            threshold=threshold,
            positive_scores=positive_top_scores,
            negative_scores=negative_top_scores,
        )
        for threshold in thresholds
    ]

    print("\nTHRESHOLD SWEEP")
    print(
        json.dumps(
            threshold_results,
            ensure_ascii=False,
            indent=2,
        ),
    )

    print("\nEVALUATION NOTES")
    print(
        "Recall@k measures whether the expected source appears "
        "within the top k results."
    )
    print(
        "MRR measures the average reciprocal rank of the expected source."
    )
    print(
        "Threshold metrics classify each query using its nearest-result score."
    )
    print(
        "These metrics describe this small synthetic dataset and do not "
        "establish a production-ready threshold."
    )


if __name__ == "__main__":
    asyncio.run(main())