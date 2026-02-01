"""Agent interface - domain layer."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any

from .state import AgentState


class AgentType(str, Enum):
    """Types of agents in the system."""
    INTENT_CLASSIFIER = "intent_classifier"
    DSL_GENERATOR = "dsl_generator"
    DIAGRAM_RENDERER = "diagram_renderer"
    PROBLEM_SOLVER = "problem_solver"


class Agent(ABC):
    """
    Base interface for all agents in the orchestration system.

    Each agent:
    - Receives AgentState
    - Performs specific task
    - Updates and returns AgentState
    """

    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.name = agent_type.value

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute agent logic and return updated state.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type})"
