"""Orchestrator interface - domain layer."""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncIterator

from .state import AgentState


class Orchestrator(ABC):
    """
    Interface for agent orchestration system.

    Orchestrator:
    - Manages workflow between agents
    - Handles routing and decision logic
    - Maintains session state
    """

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute orchestration workflow.

        Args:
            state: Initial agent state

        Returns:
            Final agent state after workflow completion
        """
        pass

    @abstractmethod
    async def stream_execute(self, state: AgentState) -> AsyncIterator[AgentState]:
        """
        Stream execution updates for real-time feedback.

        Args:
            state: Initial agent state

        Yields:
            Updated agent states during execution
        """
        pass

    @abstractmethod
    def get_workflow_graph(self) -> Dict[str, Any]:
        """
        Get workflow graph structure for visualization.

        Returns:
            Graph structure as dictionary
        """
        pass
