import asyncio
import json

from app.rag.rag_chain import RAGChain
from app.retrieval.retriever import ContentRetriever
from app.retrieval.qdrant_store import QdrantStore


async def main():

    store = QdrantStore(
        collection_name=(
            "retrieval_evaluation"
        )
    )

    retriever = ContentRetriever(
        store=store
    )

    rag = RAGChain(
        retriever=retriever
    )

    result = await rag.answer(
        query=(
            "چطور می‌توان مصرف آب "
            "کشاورزی را کاهش داد؟"
        ),
        k=3,
    )

    print("ANSWER:")
    print(result.answer)

    print("\nSOURCES:")

    print(
        json.dumps(
            [
                source.model_dump()
                for source in result.sources
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())