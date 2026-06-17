"""NLU preprocessing node."""

from __future__ import annotations

from eva_agent.nlu.preprocess import preprocess
from eva_agent.state import AgentState


def nlu_preprocessor(state: AgentState) -> dict:
    query = state.user_input_clean or state.user_input_raw
    return {"nlu": preprocess(query)}


__all__ = ["nlu_preprocessor"]
