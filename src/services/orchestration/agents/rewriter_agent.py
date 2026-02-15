import re
from typing import Any


class ProblemParser:
    def execute(self, user_input: str) -> dict[str, Any]:
        normalized = (user_input or "").strip()
        requirements = self._extract_requirements(normalized)

        return {
            "status": "success",
            "problem_statement": normalized,
            "requirements": requirements,
        }

    def infer_intent(self, requirements: list[str]) -> str:
        requirement_set = set(requirements)
        has_diagram = "diagram" in requirement_set
        has_solve = "solve" in requirement_set

        if has_diagram and has_solve:
            return "BOTH"
        if has_diagram:
            return "DIAGRAM_ONLY"
        if has_solve:
            return "SOLVE_ONLY"
        return "UNCLEAR"

    def _extract_requirements(self, user_input: str) -> list[str]:
        text = user_input.lower()

        diagram_patterns = [
            r"\bvẽ\b",
            r"\bhình\b",
            r"\bdiagram\b",
            r"\bvisual\b",
            r"\bminh họa\b",
        ]
        solve_patterns = [
            r"\bgiải\b",
            r"\btính\b",
            r"\bchứng minh\b",
            r"\bfind\b",
            r"\bsolve\b",
        ]

        requirements: list[str] = []
        if any(re.search(pattern, text) for pattern in diagram_patterns):
            requirements.append("diagram")
        if any(re.search(pattern, text) for pattern in solve_patterns):
            requirements.append("solve")

        if not requirements:
            requirements = ["diagram", "solve"]

        return requirements
