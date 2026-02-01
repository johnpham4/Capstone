"""Orchestration API - main endpoint for geometry problem solving."""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
from uuid import uuid4
from loguru import logger
import json
from datetime import datetime

from llm_src.settings import settings
from llm_src.domains.orchestration import AgentState, Intent
from llm_src.applications.orchestrators.geometry_orchestrator import GeometryOrchestrator
from llm_src.infrastructures.llm.openai_client import OpenAIClient

# Initialize orchestrator
orchestrator = GeometryOrchestrator()

# Initialize OpenAI client if API key available
if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
    openai_client = OpenAIClient(api_key=settings.OPENAI_API_KEY)
    orchestrator.set_openai_client(openai_client)
    logger.info("OpenAI client configured")
else:
    logger.warning("OPENAI_API_KEY not found - solver agent will not work")

app = FastAPI(
    title="GeoUni Orchestration API",
    description="Agent-based orchestration for geometry problem solving",
    version="2.0.0"
)


# ============ Request/Response Models ============

class OrchestrateRequest(BaseModel):
    """Request for orchestrated geometry problem solving."""
    user_input: str = Field(..., description="User's question or problem")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    intent: Optional[Intent] = Field(None, description="Pre-classified intent (optional)")
    include_diagram: bool = Field(True, description="Include diagram in response")
    include_solution: bool = Field(True, description="Include solution in response")


class OrchestrateResponse(BaseModel):
    """Response from orchestration."""
    request_id: str
    session_id: str
    intent: str
    confidence: float

    # DSL
    dsl: Optional[str] = None
    dsl_error: Optional[str] = None

    # Diagram
    has_diagram: bool = False
    diagram_url: Optional[str] = None

    # Solution
    solution: Optional[str] = None
    solution_steps: list = []
    solution_error: Optional[str] = None

    # Metadata
    execution_path: list
    errors: list
    timestamp: str


# ============ Endpoints ============

@app.post("/api/v2/orchestrate", response_model=OrchestrateResponse)
async def orchestrate_geometry_problem(request: OrchestrateRequest):
    """
    Main orchestration endpoint - automatically routes to appropriate agents.

    This endpoint:
    1. Classifies user intent
    2. Routes to appropriate agents
    3. Returns combined results (DSL, diagram, solution)

    Examples:
        - "Vẽ tam giác ABC cân tại A" → DRAW_ONLY → DSL + Diagram
        - "Chứng minh DE song song BC" → SOLVE_ONLY → Solution
        - "Vẽ và chứng minh" → DRAW_AND_SOLVE → All
    """
    try:
        request_id = str(uuid4())
        session_id = request.session_id or str(uuid4())

        logger.info(f"[{request_id}] Orchestration request: {request.user_input}")

        # Create agent state
        state = AgentState(
            session_id=session_id,
            user_input=request.user_input,
            intent=request.intent
        )
        state.add_message("user", request.user_input)

        # Execute orchestration
        result_state = await orchestrator.execute(state)

        # Build response
        response = OrchestrateResponse(
            request_id=request_id,
            session_id=session_id,
            intent=result_state.intent.value if result_state.intent else "unknown",
            confidence=result_state.confidence,
            dsl=result_state.dsl,
            dsl_error=result_state.dsl_error,
            has_diagram=result_state.diagram_bytes is not None,
            solution=result_state.solution,
            solution_steps=result_state.solution_steps,
            solution_error=result_state.solution_error,
            execution_path=result_state.execution_path,
            errors=result_state.errors,
            timestamp=datetime.now().isoformat()
        )

        # If diagram exists, provide download URL
        if result_state.diagram_bytes:
            response.diagram_url = f"/api/v2/sessions/{session_id}/diagram"

            # Cache diagram in state context for retrieval
            # In production, use Redis or S3
            if not hasattr(app.state, 'diagram_cache'):
                app.state.diagram_cache = {}
            app.state.diagram_cache[session_id] = result_state.diagram_bytes

        logger.info(f"[{request_id}] Orchestration completed: intent={response.intent}")

        return response

    except Exception as e:
        logger.exception(f"Orchestration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v2/sessions/{session_id}/diagram")
async def get_session_diagram(session_id: str):
    """
    Retrieve diagram image for a session.

    This endpoint returns the PNG diagram generated during orchestration.
    """
    try:
        if not hasattr(app.state, 'diagram_cache'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No diagram cache available"
            )

        diagram_bytes = app.state.diagram_cache.get(session_id)

        if not diagram_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No diagram found for session {session_id}"
            )

        import io
        return StreamingResponse(
            io.BytesIO(diagram_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=diagram_{session_id}.png"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to retrieve diagram: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v2/orchestrate/stream")
async def orchestrate_stream(request: OrchestrateRequest):
    """
    Stream orchestration execution for real-time updates.

    Returns SSE (Server-Sent Events) stream of agent execution steps.
    """
    async def event_generator():
        try:
            session_id = request.session_id or str(uuid4())

            state = AgentState(
                session_id=session_id,
                user_input=request.user_input,
                intent=request.intent
            )
            state.add_message("user", request.user_input)

            # Stream execution
            async for updated_state in orchestrator.stream_execute(state):
                event_data = {
                    "session_id": session_id,
                    "current_agent": updated_state.current_agent,
                    "intent": updated_state.intent.value if updated_state.intent else None,
                    "has_dsl": updated_state.dsl is not None,
                    "has_diagram": updated_state.diagram_bytes is not None,
                    "has_solution": updated_state.solution is not None,
                    "errors": updated_state.errors
                }

                yield f"data: {json.dumps(event_data)}\n\n"

            yield "data: {\"status\": \"completed\"}\n\n"

        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/api/v2/workflow/graph")
async def get_workflow_graph():
    """
    Get workflow graph structure for visualization.

    Returns the agent graph showing nodes and edges.
    """
    return orchestrator.get_workflow_graph()


@app.get("/api/v2/health")
async def health_check():
    """Health check for orchestration service."""
    return {
        "status": "healthy",
        "service": "GeoUni Orchestration API",
        "version": "2.0.0",
        "agents": {
            "intent_classifier": "active",
            "dsl_generator": "active",
            "diagram_renderer": "active",
            "problem_solver": "active" if orchestrator.solver_agent.openai_client else "inactive"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
