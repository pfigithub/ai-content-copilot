import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.rag.query_contextualizer import (
    ContextualizedQuery,
    QueryContextualizer,
)


class FakeModel:
    def __init__(self):
        self.inputs = []
        self.calls = 0

    def with_structured_output(self, _schema):
        async def invoke(input_data):
            self.calls += 1
            self.inputs.append(input_data)

            return ContextualizedQuery(
                query=(
                    "مزایای آبیاری قطره‌ای "
                    "در کاهش مصرف آب کشاورزی چیست؟"
                )
            )

        return RunnableLambda(invoke)


def test_contextualizer_rewrites_question_using_history():
    model = FakeModel()

    contextualizer = QueryContextualizer(
        model=model
    )

    history = [
        HumanMessage(
            content=(
                "چطور مصرف آب کشاورزی "
                "را کاهش دهیم؟"
            )
        ),
        AIMessage(
            content=(
                "با استفاده از آبیاری قطره‌ای "
                "و روش‌های نوین آبیاری."
            )
        ),
    ]

    query = asyncio.run(
        contextualizer.contextualize(
            question="این روش چه مزیتی دارد؟",
            history=history,
        )
    )

    assert query == (
        "مزایای آبیاری قطره‌ای "
        "در کاهش مصرف آب کشاورزی چیست؟"
    )

    assert model.calls == 1

    captured = model.inputs[0]
    
    messages = captured.to_messages()
    
    assert len(messages) == 2

    assert (
        "این روش چه مزیتی دارد؟"
        in messages[1].content
    )

    assert (
        "آبیاری قطره‌ای"
        in messages[1].content
    )
    
    assert (
        "چطور مصرف آب کشاورزی"
        in messages[1].content
    )


def test_contextualizer_handles_empty_history():
    model = FakeModel()

    contextualizer = QueryContextualizer(
        model=model
    )

    query = asyncio.run(
        contextualizer.contextualize(
            question="آبیاری قطره‌ای چیست؟",
            history=[],
        )
    )

    assert query == (
        "مزایای آبیاری قطره‌ای "
        "در کاهش مصرف آب کشاورزی چیست؟"
    )

    assert model.calls == 1


def test_contextualizer_rejects_empty_question():
    model = FakeModel()

    contextualizer = QueryContextualizer(
        model=model
    )

    try:
        asyncio.run(
            contextualizer.contextualize(
                question="",
                history=[],
            )
        )
    except ValueError as exc:
        assert str(exc) == "Question cannot be empty."
    else:
        raise AssertionError(
            "Expected ValueError"
        )