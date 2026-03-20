import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt

# Fix encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer
from src.services.diagram.optimizer import Optimizer


DEFAULT_OPTIMIZER_OPTS = {
    "epochs": 1500,
    "n_tries": 3,
    "eps": 1e-6,
    "seed": 42,
    "learning_rate": 0.01,
    "enable_chord_ndg": False,
}

INPUT_JSON_PATH = "dataset/data/train.json"
OUTPUT_DIR = "output_fixed"


def _next_output_index(output_dir: Path) -> int:
    existing_files = list(output_dir.glob("diagram_*.png"))
    if not existing_files:
        return 1

    numbers = []
    for file_path in existing_files:
        try:
            numbers.append(int(file_path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers) + 1 if numbers else 1


def test_single_problem(instruction: str, dsl_answer: str, output_path: Path) -> bool:
    print(f"Instruction: {instruction}")
    print(f"DSL: {dsl_answer}")

    try:
        dsl_lines = dsl_answer.splitlines() if "\n" in dsl_answer else [dsl_answer]
        dsl_lines = [line.strip() for line in dsl_lines if line.strip()]

        print("DSL lines:")
        for i, line in enumerate(dsl_lines, 1):
            print(f"  {i}. {line}")

        builder = DiagramBuilder(dsl_lines)

        # Print warnings if any commands were skipped
        if builder.warnings:
            print(f"\n  {len(builder.warnings)} WARNINGS (commands skipped):")
            for warning in builder.warnings:
                print(f"  {warning}")
            print()

        print(f"Points: {[p.val for p in builder.points]}")
        print(f"Instructions count: {len(builder.instructions)}")

        opts = dict(DEFAULT_OPTIMIZER_OPTS)

        optimizer = Optimizer(builder.instructions, opts, verbosity=True)
        diagram = optimizer.solve()

        print(f"\nFinal loss: {optimizer.losses}")
        print(f"Final coordinates:")
        for name, point in diagram.points.items():
            print(f"  {name}: ({point.x:.4f}, {point.y:.4f})")

        renderer = MatplotlibDiagramRenderer()
        fig, _ax = renderer.render(diagram, save=True, show=False, filename=str(output_path))

        fig.suptitle(instruction, fontsize=23, wrap=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def main():
    project_root = Path(__file__).resolve().parent
    json_path = (project_root / INPUT_JSON_PATH).resolve()
    output_dir = (project_root / OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_idx = _next_output_index(output_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"Found {len(problems)} problems")
    print(f"Starting from diagram_{start_idx:02d}.png")

    success_count = 0
    failed_problems = []  # Track failed problems

    for idx, problem in enumerate(problems, start_idx):
        instruction = problem.get("instruction", "")
        answer = problem.get("answer") or problem.get("dsl")

        if not answer:
            print(f"Skipping index {idx}: missing 'answer' or 'dsl' field")
            failed_problems.append(
                {
                    "index": idx,
                    "instruction": instruction,
                    "dsl": "",
                }
            )
            continue

        output_filename = f"diagram_{idx:02d}.png"
        output_path = output_dir / output_filename

        if test_single_problem(instruction, answer, output_path):
            success_count += 1
        else:
            failed_problems.append(
                {
                    "index": idx,
                    "instruction": instruction,
                    "dsl": answer,
                }
            )

    print(f"Completed: {success_count}/{len(problems)} diagrams generated")
    print(f"Output directory: {output_dir.absolute()}")

    # Save failed problems for review
    if failed_problems:
        failed_path = output_dir / "failed_problems.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_problems, f, ensure_ascii=False, indent=2)
        print(f"\n {len(failed_problems)} failed problems saved to: {failed_path}")
        print("Failed indices:", [p["index"] for p in failed_problems])

if __name__ == "__main__":
    main()
