import asyncio

from langchain_core.runnables import Runnable

from app.chains.summary_chain import SummaryChain
from app.models.schemas import SummaryResult


class FakeStructuredRunnable(
    Runnable,
):

    def __init__(
        self,
        result,
        captured_inputs,
    ):
        self.result = result
        self.captured_inputs = captured_inputs

    def invoke(
        self,
        input,
        config=None,
        **kwargs,
    ):
        self.captured_inputs.append(
            input
        )

        return self.result

    async def ainvoke(
        self,
        input,
        config=None,
        **kwargs,
    ):
        self.captured_inputs.append(
            input
        )

        return self.result


class FakeChatOllama:

    last_runnable = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs

    def with_structured_output(
        self,
        schema,
    ):

        assert schema is SummaryResult

        runnable = FakeStructuredRunnable(
            result=SummaryResult(
                title="Test title",
                summary="Test summary.",
                importance=70,
                category="technology",
            ),
            captured_inputs=[],
        )

        FakeChatOllama.last_runnable = runnable

        return runnable


def test_summary_chain_returns_structured_result(
    monkeypatch,
):

    monkeypatch.setattr(
        "app.chains.summary_chain.ChatOllama",
        FakeChatOllama,
    )

    chain = SummaryChain()

    result = asyncio.run(
        chain.summarize(
            "Artificial intelligence is changing content creation."
        )
    )

    assert isinstance(
        result,
        SummaryResult,
    )

    assert result.title == "Test title"
    assert result.summary == "Test summary."
    assert result.importance == 70
    assert result.category == "technology"


def test_summary_chain_passes_source_to_prompt(
    monkeypatch,
):

    monkeypatch.setattr(
        "app.chains.summary_chain.ChatOllama",
        FakeChatOllama,
    )

    chain = SummaryChain()

    source = (
        "Artificial intelligence can automate "
        "parts of content research and production."
    )

    asyncio.run(
        chain.summarize(source)
    )

    captured_input = (
        FakeChatOllama
        .last_runnable
        .captured_inputs[0]
    )

    messages = (
        captured_input.to_messages()
    )

    assert (
        messages[1].type
        == "human"
    )

    assert (
        messages[1].content
        == source
    )