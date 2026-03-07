import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from loguru import logger

from .agents import DiagramAgent, SolverAgent, RewriterAgent


Mode = Literal["diagram", "both"]


class OrchestratorState(TypedDict):
    user_input: str
    mode: Mode
    resolved_mode: Mode
    parsed: dict
    problem_statement: str
    diagram: dict
    solution: dict
    llm_mock: bool

class Orchestrator:
    def __init__(self, diagram_prompt: str):
        self.diagram_prompt = diagram_prompt
        self.diagram_agent = DiagramAgent()
        self.solver_agent = SolverAgent()
        self.rewriter = RewriterAgent()
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
        result = self.rewriter.execute(user_input=state["user_input"])
        problem_statement = result.problem_statement or state["user_input"]

        state["parsed"] = {
            "problem_statement": problem_statement,
            "mode": result.mode,
            "status": "success",
        }
        state["problem_statement"] = problem_statement
        state["resolved_mode"] = result.mode
        return state

    @staticmethod
    def _route_from_parse(state: OrchestratorState) -> str:
        return state["resolved_mode"]

    def _diagram_node(self, state: OrchestratorState) -> OrchestratorState:
        state["diagram"] = self.diagram_agent.execute(
            state["problem_statement"], self.diagram_prompt, llm_mock=state.get("llm_mock", False),
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
            "resolved_mode": "diagram",
            "parsed": {},
            "problem_statement": "",
            "diagram": {},
            "solution": {},
            "llm_mock": llm_mock,
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


    async def stream_execute(
        self,
        user_input: str,
        mode: Mode = "diagram",
        llm_mock: bool = False,
    ) -> AsyncGenerator[dict[str, Any], None]:

        try:
            result = await asyncio.to_thread(
                self.rewriter.execute, user_input,
            )
            problem = result.problem_statement or user_input
            resolved_mode = result.mode if mode == "diagram" or mode == "both" else result.mode
            resolved_mode = result.mode
        except Exception as e:
            logger.error(f"Rewriter failed: {e}")
            yield {"event": "error", "stage": "rewrite", "error": str(e)}
            return

        yield {
            "event": "rewrite",
            "problem_statement": problem,
            "mode": resolved_mode,
        }

        diagram_result: dict = {}

        yield {"event": "diagram", "status": "generating_dsl"}

        try:
            diagram_result = await asyncio.to_thread(
                self.diagram_agent.execute,
                problem,
                self.diagram_prompt,
                llm_mock,
            )
        except Exception as e:
            logger.error(f"Diagram failed: {e}")
            diagram_result = {"error": str(e), "status": "failed"}

        if diagram_result.get("status") == "failed":
            yield {"event": "diagram", "status": "failed", "error": diagram_result.get("error", "unknown")}
        else:
            yield {
                "event": "diagram",
                "status": "completed",
                "dsl": diagram_result.get("dsl"),
                "image_base64": diagram_result.get("image"),
            }

        if resolved_mode == "both":
            yield {"event": "solver", "status": "generating"}

            solve_input = problem
            dsl = diagram_result.get("dsl")
            if dsl:
                solve_input += f"\n\n[Diagram DSL: {dsl}]"

            full_solution: list[str] = []
            try:
                for token in self.solver_agent.stream_solve(solve_input):
                    full_solution.append(token)
                    yield {"event": "solver", "status": "streaming", "chunk": token}

                yield {
                    "event": "solver",
                    "status": "completed",
                    "solution": "".join(full_solution),
                }
            except Exception as e:
                logger.error(f"Solver streaming failed: {e}")
                yield {"event": "solver", "status": "failed", "error": str(e)}

        yield {
            "event": "done",
            "mode": resolved_mode,
            "has_diagram": diagram_result.get("status") != "failed",
            "has_solution": resolved_mode == "both",
        }
