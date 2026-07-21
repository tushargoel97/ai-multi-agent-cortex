from __future__ import annotations

from enum import StrEnum
from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from cortex.enums import Agents


class Intent(StrEnum):
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_QUERY = "knowledge_query"
    REASONING_TASK = "reasoning_task"
    PROMPT_CACHING = "prompt_caching"
    PRODUCT_SPECS = "product_specs"
    LOCAL_SPECIALIST = "local_specialist"
    IMAGE_GENERATION = "image_generation"
    CODING_TASK = "coding_task"
    SHOPPING = "shopping"
    BOOKING = "booking"


class RouterIntent(BaseModel):
    intent: Intent = Field(description="Required capability.")
    reasoning: str = Field(description="One-sentence routing justification.")
    agent: str | None = Field(
        default=None,
        description="Exact matching custom-agent name, otherwise null.",
    )
    local_model: str | None = Field(
        default=None,
        description="Exact matching local-specialist model ID, otherwise null.",
    )
    complexity: Literal["simple", "standard", "complex"] = "standard"


class ChatState(MessagesState):
    summary: str
    summary_upto: int
    spec_draft: str
    spec_gap: str


INTENT_TO_NODE = {
    Intent.GENERAL_CHAT: "generalist",
    Intent.KNOWLEDGE_QUERY: "researcher",
    Intent.REASONING_TASK: "reasoner",
    Intent.PROMPT_CACHING: "prompt_cacher",
    Intent.PRODUCT_SPECS: "specialist",
    Intent.LOCAL_SPECIALIST: "specialist",
    Intent.IMAGE_GENERATION: "imagegen",
    Intent.CODING_TASK: "coder",
    Intent.SHOPPING: "shopping",
    Intent.BOOKING: "booking",
}
NODE_TO_INTENT = {node: intent.value for intent, node in INTENT_TO_NODE.items()}
NODE_TO_INTENT[Agents.DEBUGGER.value] = Intent.CODING_TASK.value
NODE_TO_INTENT["specialist"] = Intent.PRODUCT_SPECS.value
