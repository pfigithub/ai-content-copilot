import asyncio
import json

from app.retrieval.document_splitter import (
    split_document,
)
from app.retrieval.qdrant_store import (
    QdrantStore,
)


COLLECTION_NAME = "retrieval_evaluation"


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


EVALUATION_CASES = [
    {
        "query": "چطور می‌توان مصرف آب کشاورزی را کاهش داد؟",
        "expected_source": "agriculture.txt",
    },
    {
        "query": "هوش مصنوعی چگونه به تولید محتوا کمک می‌کند؟",
        "expected_source": "ai.txt",
    },
    {
        "query": "چه عواملی روی سرمایه‌گذاری و فعالیت اقتصادی تأثیر می‌گذارند؟",
        "expected_source": "economics.txt",
    },
    {
        "query": "چگونه می‌توان عادت مطالعه بهتری ایجاد کرد؟",
        "expected_source": "education.txt",
    },
    {
        "query": "پروتکل‌ها در شبکه‌های کامپیوتری چه نقشی دارند؟",
        "expected_source": "network.txt",
    },
    {
        "query": "چطور می‌توان عملکرد ورزشکاران را بهبود داد؟",
        "expected_source": "sports.txt",
    },
]


def build_documents():

    documents = []

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


async def main():

    store = QdrantStore(
        collection_name=COLLECTION_NAME,
    )

    if store.client.collection_exists(
        collection_name=COLLECTION_NAME
    ):
        store.client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    documents = build_documents()

    ids = store.add_documents(
        documents
    )

    print(
        "documents:",
        len(documents),
    )

    print(
        "inserted:",
        len(ids),
    )

    total = 0
    top_1_hits = 0

    for case in EVALUATION_CASES:

        total += 1

        results = await (
            store.similarity_search_with_score(
                query=case["query"],
                k=3,
            )
        )

        top_source = None

        if results:
            top_source = (
                results[0][0]
                .metadata
                .get("source")
            )

        top_1_match = (
            top_source
            == case["expected_source"]
        )

        if top_1_match:
            top_1_hits += 1

        print("\nQUERY:")
        print(case["query"])

        output = []

        for rank, (
            document,
            score,
        ) in enumerate(
            results,
            start=1,
        ):

            output.append(
                {
                    "rank": rank,
                    "source": (
                        document.metadata.get(
                            "source"
                        )
                    ),
                    "score": score,
                    "content": (
                        document.page_content
                    ),
                }
            )

        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            "expected:",
            case["expected_source"],
        )

        print(
            "top_1_match:",
            top_1_match,
        )

    print("\nEVALUATION SUMMARY")
    print("total:", total)
    print("top_1_hits:", top_1_hits)

    accuracy = (
        top_1_hits / total
        if total
        else 0
    )

    print(
        "top_1_accuracy:",
        f"{accuracy:.2%}",
    )


if __name__ == "__main__":
    asyncio.run(main())