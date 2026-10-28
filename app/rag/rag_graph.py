import logging
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.rag.tools import search_internal_documents, search_web
from app.services.guardrail_service import OutputGuardrail, SecurityViolationError

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """The state of the agent, maintaining the history of messages."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class RAGGraph:
    """An autonomous Agent that decides when to use tools and when to chat."""
    
    def __init__(self, output_guardrail: OutputGuardrail | None = None):
        self.output_guardrail = output_guardrail or OutputGuardrail()
        
        # 1. Initialize the LLM
        self.model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        
        # 2. Bind tools to the model
        self.tools = [search_internal_documents, search_web]
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        # 3. Define the Agent's Brain (System Prompt)
        self.system_prompt = SystemMessage(
            content=(
                "You are an intelligent, professional AI assistant.\n"
                "RULES:\n"
                "1. For normal conversations, answer directly.\n"
                "2. If the user asks about specific personal facts or uploaded PDFs, call the 'search_internal_documents' tool.\n"
                "3. If the user asks about current events, weather, or real-time info, call the 'search_web' tool.\n"
                "4. CRITICAL: When providing the final answer, behave like a human expert. Give ONLY the answer to the user's question. Do not explain your thought process, do not list your rules, and do not explain how you found the information.\n"
                "5. LANGUAGE RULE: Always reply in the exact same language the user used. "
                "If you cannot detect the language or understand the question, reply EXACTLY with:\n"
                "'Sorry, I couldn't detect the language or understand the question. Could you please clarify?'\n"
                "'متاسفم، زبان شما را تشخیص ندادم یا متوجه سوال نشدم. لطفاً واضح‌تر بپرسید.'"
            )
        )

        # 4. Build the Graph
        builder = StateGraph(AgentState)
        
        # Nodes
        builder.add_node("agent", self.call_model)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_node("guardrail", self.run_guardrail)
        
        # Edges
        builder.add_edge(START, "agent")
        
        # The agent decides: use a tool (continue) or finish (end)
        builder.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": "guardrail"
            }
        )
        
        # After using a tool, go back to the agent to read the result and answer
        builder.add_edge("tools", "agent")
        builder.add_edge("guardrail", END)
        
        self.graph = builder.compile()

    async def call_model(self, state: AgentState) -> dict:
        """Call the LLM with the current context and tools."""
        messages = state["messages"]
        
        # Ensure system prompt is always applied
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [self.system_prompt] + list(messages)
            
        response = await self.model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(self, state: AgentState) -> str:
        """Check if the agent decided to call a tool."""
        last_message = state["messages"][-1]
        
        if last_message.tool_calls:
            return "continue"
        return "end"

    async def run_guardrail(self, state: AgentState) -> dict:
        """Final check before returning the output to the user."""
        last_message = state["messages"][-1]
        
        if not last_message.content:
            return {"messages": []}
            
        try:
            await self.output_guardrail.verify_safe_output(last_message.content)
        except SecurityViolationError as exc:
            logger.warning("Output guardrail triggered", exc_info=True)
            # Overwrite the malicious message with a safe fallback
            safe_message = AIMessage(
                content="The response was blocked due to security policy violations.",
                id=last_message.id
            )
            return {"messages": [safe_message]}
            
        return {"messages": []}