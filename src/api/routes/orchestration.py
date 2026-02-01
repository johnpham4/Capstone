"""Orchestration API - main endpoint for geometry problem solving."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, Literal
from uuid import uuid4
from loguru import logger
import json
from datetime import datetime

from src.config.settings.base import settings
from src.models.domain.orchestration import AgentState, Intent
from src.models.schemas import OrchestrateRequest, OrchestrateResponse
from src.services.orchestrators.geometry_orchestrator import GeometryOrchestrator
from src.infrastructures.llm.openai_client import OpenAIClient

router = APIRouter()

orchestrator = GeometryOrchestrator()


@router.post("/api/v2/orchestrate", response_model=OrchestrateResponse)
async def orchestrate_geometry_problem(request: OrchestrateRequest):
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
            solution=result_state.solution,
            solution_error=result_state.solution_error,
            steps_executed=getattr(result_state, 'execution_path', None),
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


@router.get("/api/v2/sessions/{session_id}/diagram")
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


@router.post("/api/v2/orchestrate/stream")
async def orchestrate_stream(request: OrchestrateRequest):
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


@router.get("/api/v2/workflow/graph")
async def get_workflow_graph():
    """
    Get workflow graph structure for visualization.

    Returns the agent graph showing nodes and edges.
    """
    return orchestrator.get_workflow_graph()


@router.get("/api/v2/health")
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


