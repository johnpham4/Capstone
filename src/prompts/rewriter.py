REWRITER_SYSTEM_PROMPT: str = """\
Extract the geometry problem from the user message and decide the intent.

Rules:
- problem_statement: the clean geometry problem (keep original language).
- mode: "diagram" if user only wants a diagram, "both" if user also wants a solution. Default "diagram".

{format_instructions}
"""
