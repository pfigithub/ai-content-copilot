from collections.abc import Sequence

from langchain_core.messages import AnyMessage, RemoveMessage


MAX_SHORT_TERM_MESSAGES = 20


def get_messages_to_remove(
    messages: Sequence[AnyMessage],
    max_messages: int = MAX_SHORT_TERM_MESSAGES,
) -> list[RemoveMessage]:
    if max_messages <= 0:
        raise ValueError(
            "max_messages must be greater than zero."
        )

    if len(messages) <= max_messages:
        return []

    old_messages = messages[:-max_messages]

    return [
        RemoveMessage(id=message.id)
        for message in old_messages
        if message.id is not None
    ]