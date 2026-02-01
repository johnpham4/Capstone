"""Geometry Orchestrator - coordinates agents to solve geometry problems."""

from typing import Dict, Any, AsyncIterator
from loguru import logger

from src.models.domain.orchestration import Orchestrator, AgentState, Intent
from src.services.agents import (
    IntentClassifierAgent,
    DSLGeneratorAgent,
    DiagramRendererAgent,
    ProblemSolverAgent
)


class GeometryOrchestrator(Orchestrator):
    """
    Orchestrates geometry problem solving workflow.

    Workflow:
    1. Classify intent → IntentAgent
    2. Route based on intent:
       - DRAW_ONLY → DSL → Diagram
       - SOLVE_ONLY → Solver (uses existing DSL if available)
       - DRAW_AND_SOLVE → DSL → Diagram → Solver (parallel where possible)
       - CLARIFY → Return clarification request
    """

    def __init__(self):
        # Initialize agents
        self.intent_agent = IntentClassifierAgent()
        self.dsl_agent = DSLGeneratorAgent()
        self.diagram_agent = DiagramRendererAgent()
        self.solver_agent = ProblemSolverAgent()

        logger.info("GeometryOrchestrator initialized")

    def set_openai_client(self, client):
        """Inject OpenAI client to solver agent."""
        self.solver_agent.set_openai_client(client)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute orchestration workflow."""
        logger.info(f"Orchestration started for session: {state.session_id}")

        try:
            # Step 1: Classify intent (if not already classified)
            if not state.intent:
                state = await self.intent_agent.execute(state)

                if state.intent == Intent.UNKNOWN:
                    logger.warning("Unknown intent, returning state")
                    return state

            # Step 2: Route based on intent
            if state.intent == Intent.DRAW_ONLY:
                state = await self._execute_draw_workflow(state)

            elif state.intent == Intent.SOLVE_ONLY:
                state = await self._execute_solve_workflow(state)

            elif state.intent == Intent.DRAW_AND_SOLVE:
                state = await self._execute_full_workflow(state)

            elif state.intent == Intent.CLARIFY:
                # Add clarification message
                state.add_message(
                    "assistant",
                    "Xin lỗi, tôi chưa hiểu rõ yêu cầu của bạn. "
                    "Bạn muốn vẽ hình hay giải bài toán?"
                )

            logger.info(
                f"Orchestration completed. Path: {' -> '.join(state.execution_path)}"
            )

        except Exception as e:
            error_msg = f"Orchestration failed: {str(e)}"
            state.add_error(error_msg)
            logger.error(error_msg, exc_info=True)

        return state

    async def _execute_draw_workflow(self, state: AgentState) -> AgentState:
        """Execute: DSL generation → Diagram rendering."""
        logger.info("Executing DRAW_ONLY workflow")

        # Generate DSL
        state = await self.dsl_agent.execute(state)

        # If DSL generation failed, stop
        if state.dsl_error:
            return state

        # Render diagram
        state = await self.diagram_agent.execute(state)

        return state

    async def _execute_solve_workflow(self, state: AgentState) -> AgentState:
        """Execute: Problem solving (with existing context)."""
        logger.info("Executing SOLVE_ONLY workflow")

        # Check if we have DSL context, if not generate it
        if not state.dsl:
            logger.info("No DSL context, generating...")
            state = await self.dsl_agent.execute(state)

        # Solve problem
        state = await self.solver_agent.execute(state)

        return state

    async def _execute_full_workflow(self, state: AgentState) -> AgentState:
        """Execute: DSL → Diagram + Solve (parallel where possible)."""
        logger.info("Executing DRAW_AND_SOLVE workflow")

        # Generate DSL first
        state = await self.dsl_agent.execute(state)

        if state.dsl_error:
            return state

        # Render diagram and solve in parallel
        # For now, sequential (can optimize later with asyncio.gather)
        state = await self.diagram_agent.execute(state)
        state = await self.solver_agent.execute(state)

        return state

    async def stream_execute(self, state: AgentState) -> AsyncIterator[AgentState]:
        """Stream execution updates (for real-time feedback)."""
        # Yield initial state
        yield state

        # Classify intent
        if not state.intent:
            state = await self.intent_agent.execute(state)
            yield state

        # Route and yield after each agent
        if state.intent == Intent.DRAW_ONLY:
            state = await self.dsl_agent.execute(state)
            yield state

            state = await self.diagram_agent.execute(state)
            yield state

        elif state.intent == Intent.SOLVE_ONLY:
            if not state.dsl:
                state = await self.dsl_agent.execute(state)
                yield state

            state = await self.solver_agent.execute(state)
            yield state

        elif state.intent == Intent.DRAW_AND_SOLVE:
            state = await self.dsl_agent.execute(state)
            yield state

            state = await self.diagram_agent.execute(state)
            yield state

            state = await self.solver_agent.execute(state)
            yield state

    def get_workflow_graph(self) -> Dict[str, Any]:
        """Get workflow graph structure."""
        return {
            "nodes": [
                {"id": "intent", "type": "classifier", "agent": "IntentAgent"},
                {"id": "dsl", "type": "generator", "agent": "DSLAgent"},
                {"id": "diagram", "type": "renderer", "agent": "DiagramAgent"},
                {"id": "solver", "type": "solver", "agent": "SolverAgent"}
            ],
            "edges": [
                {"from": "start", "to": "intent"},
                {"from": "intent", "to": "dsl", "condition": "DRAW_*"},
                {"from": "intent", "to": "solver", "condition": "SOLVE_ONLY"},
                {"from": "dsl", "to": "diagram"},
                {"from": "diagram", "to": "solver", "condition": "DRAW_AND_SOLVE"},
                {"from": "diagram", "to": "end", "condition": "DRAW_ONLY"},
                {"from": "solver", "to": "end"}
            ]
        }
