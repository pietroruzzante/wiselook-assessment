"""Builds and compiles the assessment StateGraph.

NOTE: MemorySaver is single-process, in-memory — fine for this test.
AsyncPostgresSaver is a one-line swap for production persistence (see
README) since both implement the same BaseCheckpointSaver interface.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.domain.state import AssessmentState
from app.graph.edges import route_after_score, route_after_sufficiency
from app.graph.nodes import (
    ask_followup,
    ask_question,
    assess_sufficiency,
    finalize,
    score_dimension,
    wait_for_input,
)


def build_graph() -> CompiledStateGraph[AssessmentState, None, AssessmentState, AssessmentState]:
    graph = StateGraph(AssessmentState)

    graph.add_node("ask_question", ask_question)
    graph.add_node("wait_for_input", wait_for_input)
    graph.add_node("assess_sufficiency", assess_sufficiency)
    graph.add_node("ask_followup", ask_followup)
    graph.add_node("score_dimension", score_dimension)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "ask_question")
    graph.add_edge("ask_question", "wait_for_input")
    graph.add_edge("wait_for_input", "assess_sufficiency")
    graph.add_conditional_edges(
        "assess_sufficiency",
        route_after_sufficiency,
        {"ask_followup": "ask_followup", "score_dimension": "score_dimension"},
    )
    graph.add_edge("ask_followup", "wait_for_input")
    graph.add_conditional_edges(
        "score_dimension",
        route_after_score,
        {"ask_question": "ask_question", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
