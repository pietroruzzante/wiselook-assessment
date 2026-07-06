"""Shared FastAPI dependencies: the compiled graph singleton and the
per-session LangGraph config helper."""

from __future__ import annotations

from functools import lru_cache
from typing import Any
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from app.domain.state import AssessmentState
from app.graph.builder import build_graph


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph[AssessmentState, None, AssessmentState, AssessmentState]:
    """Compiled graph is stateless itself (all session state lives in the
    checkpointer), so one instance safely serves every request."""
    return build_graph()


def thread_config(session_id: UUID) -> dict[str, Any]:
    """The LangGraph RunnableConfig that scopes a call to one session's thread."""
    return {"configurable": {"thread_id": str(session_id)}}
