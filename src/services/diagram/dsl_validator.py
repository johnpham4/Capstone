# import re
# from loguru import logger


# def validate_and_fix_dsl(dsl: str) -> str:
#     """Clean, validate and fix LLM-generated DSL. Returns sanitized DSL string."""
#     lines = [line.strip() for line in dsl.splitlines() if line.strip()]
#     if not lines:
#         return dsl

#     fixed_lines: list[str] = []
#     defined_points: set[str] = set()
#     seen: set[str] = set()

#     # ── Pass 1: collect shape-defined points ────────────────
#     for line in lines:
#         _collect_shape_points(line, defined_points)

#     # ── Pass 2: collect explicitly defined points ───────────
#     for line in lines:
#         _collect_define_points(line, defined_points)

#     # ── Pass 3: validate each line ──────────────────────────
#     removed: list[str] = []
#     for line in lines:
#         # Fix unbalanced parens
#         line = _fix_parens(line)
#         if not line:
#             continue

#         # Deduplicate
#         if line in seen:
#             continue
#         seen.add(line)

#         # Validate specific constructs
#         issue = _check_line(line, defined_points)
#         if issue:
#             removed.append(f"{line}  # {issue}")
#             continue

#         fixed_lines.append(line)

#     if removed:
#         logger.warning(f"DSL validator removed {len(removed)} lines:")
#         for r in removed:
#             logger.warning(f"  - {r}")

#     result = "\n".join(fixed_lines)
#     if result != dsl.strip():
#         logger.info(f"DSL sanitized: {len(lines)} -> {len(fixed_lines)} lines")
#     return result


# def _fix_parens(line: str) -> str:
#     """Fix unbalanced parentheses by closing or trimming."""
#     opens = line.count("(")
#     closes = line.count(")")
#     if opens > closes:
#         # Truncated output — close remaining parens
#         line += ")" * (opens - closes)
#     elif closes > opens:
#         # Extra closing parens — trim from right
#         while line.count(")") > line.count("("):
#             idx = line.rfind(")")
#             line = line[:idx] + line[idx + 1 :]
#     return line.strip()


# _SHAPE_COMMANDS = {"triangle", "rectangle", "square", "parallelogram", "rhombus", "trapezoid", "quadrilateral"}


# def _collect_shape_points(line: str, defined: set[str]) -> None:
#     """Extract point names from shape declarations like (rectangle (A B C D))."""
#     tokens = line.replace("(", " ").replace(")", " ").split()
#     if not tokens:
#         return
#     head = tokens[0].lower()
#     if head in _SHAPE_COMMANDS:
#         for tok in tokens[1:]:
#             if tok[0].isupper() and tok.isalnum():
#                 defined.add(tok)


# def _collect_define_points(line: str, defined: set[str]) -> None:
#     """Extract point names from (define X point ...)."""
#     tokens = line.replace("(", " ").replace(")", " ").split()
#     if len(tokens) >= 3 and tokens[0].lower() == "define" and tokens[2].lower() == "point":
#         defined.add(tokens[1])


# def _check_line(line: str, defined: set[str]) -> str | None:
#     """Return error message if line is invalid, None if OK."""
#     tokens = line.replace("(", " ").replace(")", " ").split()
#     if not tokens:
#         return "empty line"

#     head = tokens[0].lower()

#     # ── inter-ll validation ─────────────────────────────────
#     if "inter-ll" in tokens or "inter_ll" in tokens:
#         # Extract the 4 point args after inter-ll
#         try:
#             idx = next(i for i, t in enumerate(tokens) if t in ("inter-ll", "inter_ll"))
#             args = [t for t in tokens[idx + 1 :] if t[0].isupper() and t.isalnum()]
#         except StopIteration:
#             args = []

#         if len(args) < 4:
#             return f"inter-ll needs 4 points, got {len(args)}: {args}"

#         p1, p2, p3, p4 = args[:4]
#         # Two lines must not be identical
#         line1 = frozenset([p1, p2])
#         line2 = frozenset([p3, p4])
#         if line1 == line2:
#             return f"inter-ll degenerate: both lines are {p1}{p2}"
#         # A line defined by the same point twice is degenerate
#         if p1 == p2 or p3 == p4:
#             return f"inter-ll degenerate: line has same point twice"
#         # Lines sharing a point → intersection is trivially that point
#         shared = line1 & line2
#         if shared:
#             return f"inter-ll trivial: lines share point {next(iter(shared))}"

#     # ── Check all uppercase tokens are defined ──────────────
#     point_refs = [t for t in tokens if t[0].isupper() and t.isalnum() and len(t) <= 3]
#     # Skip the head command and type keywords
#     skip = {"point", "line", "segment", "circle", "midpoint", "projection",
#             "bisector", "inter-ll", "inter_ll", "coords", "perp-bisector"}
#     for ref in point_refs:
#         if ref.lower() in skip:
#             continue
#         if ref not in defined:
#             # If this is a define line, the new point is being defined now
#             if head == "define" and len(tokens) >= 2 and tokens[1] == ref:
#                 continue
#             return f"undefined point '{ref}'"

#     return None
