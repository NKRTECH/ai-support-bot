"""
Graph state schema for the multi-agent support bot.

Defines the shared state that flows through all graph nodes.
Uses Pydantic for validation and Annotated reducers for
controlling how node outputs merge into the state.
"""

import operator
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class SupportState(BaseModel):
    """Shared state for the support agent graph.

    Each field is either replaced (default) or appended (with operator.add)
    when a node returns an update.
    """

    # Conversation history — appended by each node that adds messages
    messages: Annotated[list[BaseMessage], operator.add] = Field(
        default_factory=list
    )

    # Intent classification (set by triage node)
    intent: str = ""
    confidence: float = 0.0
    entities: dict[str, Any] = Field(default_factory=dict)

    # RAG context (set by faq node)
    retrieved_context: str = ""

    # Tool execution results (set by action node)
    tool_results: list[str] = Field(default_factory=list)

    # Draft and final responses
    draft_response: str = ""
    final_response: str = ""

    # Quality check (set by quality node)
    quality_score: float = 0.0
    retry_count: int = 0

    # Human-in-the-loop (set by action node)
    needs_human_approval: bool = False
    is_approved: bool = False
