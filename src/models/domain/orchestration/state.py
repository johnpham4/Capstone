"""Agent state management - domain entity."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class Intent(str, Enum):
    """User intent classification."""
    DRAW_ONLY = "draw_only"           # Chỉ vẽ diagram
    SOLVE_ONLY = "solve_only"         # Chỉ giải bài toán
    DRAW_AND_SOLVE = "draw_and_solve" # Vẽ + giải
    CLARIFY = "clarify"                # Cần làm rõ
    UNKNOWN = "unknown"


@dataclass
class Message:
    """Chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """
    Shared state across all agents in orchestration workflow.

    This is the single source of truth that flows through the agent graph.
    """
    # Input
    session_id: str
    user_input: str

    # Intent classification
    intent: Optional[Intent] = None
    confidence: float = 0.0

    # DSL generation
    dsl: Optional[str] = None
    dsl_error: Optional[str] = None

    # Diagram rendering
    diagram_bytes: Optional[bytes] = None
    diagram_error: Optional[str] = None

    # Problem solving
    solution: Optional[str] = None
    solution_steps: List[str] = field(default_factory=list)
    solution_error: Optional[str] = None

    # Session & history
    history: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    current_agent: Optional[str] = None
    execution_path: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add message to history."""
        self.history.append(Message(role=role, content=content))

    def add_execution_step(self, agent_name: str) -> None:
        """Track agent execution path."""
        self.execution_path.append(agent_name)
        self.current_agent = agent_name

    def add_error(self, error: str) -> None:
        """Add error to error list."""
        self.errors.append(error)

    def get_last_user_message(self) -> Optional[str]:
        """Get last user message from history."""
        for msg in reversed(self.history):
            if msg.role == "user":
                return msg.content
        return None

    def get_context_for_solver(self) -> str:
        """Build context string for solver agent."""
        parts = []

        if self.dsl:
            parts.append(f"Geometry DSL:\n{self.dsl}")

        if self.history:
            recent = self.history[-3:]  # Last 3 messages
            history_str = "\n".join([f"{m.role}: {m.content}" for m in recent])
            parts.append(f"Conversation:\n{history_str}")

        return "\n\n".join(parts)
