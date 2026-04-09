import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from .agents import DiagramAgent, SolverAgent


Mode = Literal["diagram", "solve", "both"]


class OrchestratorState(TypedDict):
    user_input: str
    mode: Mode
    resolved_mode: Mode
    problem_statement: str
    diagram: dict
    solution: dict
    llm_mock: bool

class Orchestrator:
    def __init__(self, diagram_prompt: str):
        self.diagram_prompt = diagram_prompt
        self.diagram_agent = DiagramAgent()
        self.solver_agent = SolverAgent()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("parse", self._parse_node)
        workflow.add_node("diagram", self._diagram_node)
        workflow.add_node("solve", self._solve_node)

        workflow.set_entry_point("parse")
        workflow.add_conditional_edges(
            "parse",
            self._route_from_parse,
            {
                "diagram": "diagram",
                "solve": "solve",
                "both": "diagram",
            },
        )
        workflow.add_conditional_edges(
            "diagram",
            lambda state: "solve" if state["resolved_mode"] == "both" else END,
            {
                "solve": "solve",
                END: END,
            },
        )
        workflow.add_edge("solve", END)
        return workflow.compile()

    def _parse_node(self, state: OrchestratorState) -> OrchestratorState:
        state["resolved_mode"] = state.get("mode", "diagram")
        state["problem_statement"] = state.get("user_input", "")
        return state

    @staticmethod
    def _route_from_parse(state: OrchestratorState) -> str:
        return state["resolved_mode"]

    def _diagram_node(self, state: OrchestratorState) -> OrchestratorState:
        state["diagram"] = self.diagram_agent.execute(
            state["problem_statement"],
            self.diagram_prompt,
            llm_mock=state.get("llm_mock", False),
        )
        return state

    def _solve_node(self, state: OrchestratorState) -> OrchestratorState:
        solve_input = state["problem_statement"]
        if state.get("diagram") and state["diagram"].get("dsl"):
            solve_input += f"\n\n[Diagram DSL: {state['diagram']['dsl']}]"
        state["solution"] = self.solver_agent.execute(solve_input)
        return state

    async def execute(
        self,
        user_input: str,
        mode: Mode = "diagram",
        llm_mock: bool = False,
    ) -> dict:
        initial_state: OrchestratorState = {
            "user_input": user_input,
            "mode": mode,
            "resolved_mode": mode,
            "problem_statement": user_input,
            "diagram": {},
            "solution": {},
            "llm_mock": llm_mock,
        }

        final_state = await asyncio.to_thread(self.workflow.invoke, initial_state)

        result: dict = {"mode": final_state.get("resolved_mode", mode)}
        if final_state.get("diagram"):
            result["diagram"] = final_state["diagram"]
        if final_state.get("solution"):
            result["solution"] = final_state["solution"]

        return result


    async def stream_execute(
        self,
        user_input: str,
        mode: Mode = "diagram",
        llm_mock: bool = False,
    ) -> AsyncGenerator[dict[str, Any], None]:
        pass