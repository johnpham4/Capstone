"""Intent classification agent - determines user's goal."""

import re
from typing import Dict, Any
from loguru import logger

from src.models.domain.orchestration import Agent, AgentType, AgentState, Intent


class IntentClassifierAgent(Agent):
    """
    Classify user intent from input text.

    Intents:
    - DRAW_ONLY: "Vẽ tam giác ABC", "Hãy vẽ..."
    - SOLVE_ONLY: "Chứng minh...", "Tính...", "Tìm..."
    - DRAW_AND_SOLVE: "Vẽ và chứng minh...", "Vẽ hình và tính..."
    - CLARIFY: Unclear or needs more info
    """

    # Keywords for classification
    DRAW_KEYWORDS = [
        "vẽ", "vẽ hình", "biểu diễn", "minh họa", "diagram",
        "hình vẽ", "sketch"
    ]

    SOLVE_KEYWORDS = [
        "chứng minh", "tính", "tìm", "giải", "cm", "tính toán",
        "xác định", "calculate", "prove", "find", "solve"
    ]

    def __init__(self):
        super().__init__(AgentType.INTENT_CLASSIFIER)
        self.confidence_threshold = 0.7

    async def execute(self, state: AgentState) -> AgentState:
        """Classify user intent."""
        state.add_execution_step(self.name)

        try:
            user_input = state.user_input.lower()

            # Check for draw keywords
            has_draw = any(keyword in user_input for keyword in self.DRAW_KEYWORDS)

            # Check for solve keywords
            has_solve = any(keyword in user_input for keyword in self.SOLVE_KEYWORDS)

            # Classify
            if has_draw and has_solve:
                intent = Intent.DRAW_AND_SOLVE
                confidence = 0.9
            elif has_draw:
                intent = Intent.DRAW_ONLY
                confidence = 0.85
            elif has_solve:
                # If solving but no geometry DSL context, might need drawing first
                if not state.dsl and not self._has_geometry_description(user_input):
                    intent = Intent.CLARIFY
                    confidence = 0.6
                else:
                    intent = Intent.SOLVE_ONLY
                    confidence = 0.8
            elif self._is_geometry_description(user_input):
                # Pure geometry description without explicit "draw" → assume DRAW_ONLY
                intent = Intent.DRAW_ONLY
                confidence = 0.75
            else:
                intent = Intent.CLARIFY
                confidence = 0.5

            state.intent = intent
            state.confidence = confidence

            logger.info(
                f"Intent classified: {intent.value} "
                f"(confidence: {confidence:.2f})"
            )

        except Exception as e:
            error_msg = f"Intent classification failed: {str(e)}"
            state.add_error(error_msg)
            state.intent = Intent.UNKNOWN
            state.confidence = 0.0
            logger.error(error_msg)

        return state

    def _is_geometry_description(self, text: str) -> bool:
        """Check if text describes geometry (shapes, points, etc)."""
        geometry_terms = [
            "tam giác", "hình vuông", "đường tròn", "điểm",
            "đường thẳng", "đoạn thẳng", "góc", "cạnh",
            "triangle", "square", "circle", "point", "line"
        ]
        return any(term in text for term in geometry_terms)

    def _has_geometry_description(self, text: str) -> bool:
        """Check if text has enough geometry info to work with."""
        # Count geometry entities
        entities = re.findall(r'\b[A-Z]\b', text)  # Single capital letters (points)
        return len(entities) >= 3 or self._is_geometry_description(text)

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "type": self.agent_type.value,
            "confidence_threshold": self.confidence_threshold,
            "draw_keywords": self.DRAW_KEYWORDS,
            "solve_keywords": self.SOLVE_KEYWORDS
        }
