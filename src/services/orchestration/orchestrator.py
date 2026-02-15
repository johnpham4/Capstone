import asyncio
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from .agents import DiagramAgent, SolverAgent, ProblemParser


Mode = Literal["auto", "diagram", "solve", "both"]
Intent = Literal["DIAGRAM_ONLY", "SOLVE_ONLY", "BOTH", "UNCLEAR"]


class OrchestratorState(TypedDict):
    user_input: str
    mode: Mode
    resolved_mode: Literal["diagram", "solve", "both"]
    parsed: dict
    problem_statement: str
    diagram: dict
    solution: dict

class Orchestrator:
    def __init__(self, diagram_prompt: str):
        self.diagram_prompt = diagram_prompt
        self.diagram_agent = DiagramAgent()
        self.solver_agent = SolverAgent()
        self.parser = ProblemParser()
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
        parsed = self.parser.execute(state["user_input"])
        problem_statement = parsed.get("problem_statement") or state["user_input"]

        resolved_mode = state["mode"]
        if state["mode"] == "auto":
            intent: Intent = self.parser.infer_intent(parsed.get("requirements", []))
            resolved_mode = self._intent_to_mode(intent)

        state["parsed"] = {
            "problem_statement": problem_statement,
            "requirements": parsed.get("requirements", []),
            "status": parsed.get("status", "success"),
        }
        state["problem_statement"] = problem_statement
        state["resolved_mode"] = resolved_mode
        return state

    @staticmethod
    def _route_from_parse(state: OrchestratorState) -> Literal["diagram", "solve", "both"]:
        return state["resolved_mode"]

    def _diagram_node(self, state: OrchestratorState) -> OrchestratorState:
        state["diagram"] = self.diagram_agent.execute(state["problem_statement"], self.diagram_prompt)
        return state

    def _solve_node(self, state: OrchestratorState) -> OrchestratorState:
        solve_input = state["problem_statement"]
        if state.get("diagram") and state["diagram"].get("dsl"):
            solve_input += f"\n\n[Diagram DSL: {state['diagram']['dsl']}]"
        state["solution"] = self.solver_agent.execute(solve_input)
        return state

    async def execute(self, user_input: str, mode: Mode = "auto") -> dict:
        initial_state: OrchestratorState = {
            "user_input": user_input,
            "mode": mode,
            "resolved_mode": "both",
            "parsed": {},
            "problem_statement": "",
            "diagram": {},
            "solution": {},
        }

        final_state = await asyncio.to_thread(self.workflow.invoke, initial_state)

        result: dict = {
            "parsed": final_state.get("parsed", {}),
            "mode": final_state.get("resolved_mode", mode),
        }
        if final_state.get("diagram"):
            result["diagram"] = final_state["diagram"]
        if final_state.get("solution"):
            result["solution"] = final_state["solution"]

        return result

    @staticmethod
    def _intent_to_mode(intent: Intent) -> Literal["diagram", "solve", "both"]:
        if intent == "DIAGRAM_ONLY":
            return "diagram"
        if intent == "SOLVE_ONLY":
            return "solve"
        return "both"
