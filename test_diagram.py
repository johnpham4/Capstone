import io
import json
import re
import sys
import traceback
from pathlib import Path
import sys
import io
import argparse
import yaml

# Fix encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.optimizer import Optimizer
from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_render_config(project_root: Path, config_path: str) -> dict:
    resolved_config_path = _resolve_path(project_root, config_path)

    if not resolved_config_path.exists():
        return {}

    with open(resolved_config_path, "r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}

    if not isinstance(parsed, dict):
        return {}

    parameters = parsed.get("parameters", {})
    return parameters if isinstance(parameters, dict) else {}

def test_single_problem(instruction, dsl_answer, output_path):
    print(f"Instruction: {instruction}")
    print(f"DSL: {dsl_answer}")

    try:
        dsl_lines = dsl_answer.split('\n') if '\n' in dsl_answer else [dsl_answer]
        dsl_lines = [line.strip() for line in dsl_lines if line.strip()]
        
        print(f"DSL lines:")
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
        # print(f"Instructions: {builder.instructions}")
        opts = {
            'epochs': 1500,
            'n_tries': 3,
            'eps': 1e-6,
            'seed': 42,
            'learning_rate': 0.01,
            'enable_chord_ndg': False,
        }

        optimizer = Optimizer(builder.instructions, opts, verbosity=True)
        diagram = optimizer.solve()

        print(f"\nFinal loss: {optimizer.losses}")
        print(f"Final coordinates:")
        print(f"\nFinal loss: {optimizer.losses}")
        print(f"Final coordinates:")
        for name, point in diagram.points.items():
            print(f"  {name}: ({point.x:.4f}, {point.y:.4f})")

        renderer = MatplotlibDiagramRenderer()
        fig, _ax = renderer.render(diagram, save=True, show=False, filename=str(output_path))

        fig.suptitle(instruction, fontsize=23, wrap=True)
        fig.suptitle(instruction, fontsize=23, wrap=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error: {e}")

        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Render diagrams from DSL dataset")
    parser.add_argument("--config", default="configs/diagram_render.yaml", help="YAML config path")
    parser.add_argument("--input", default=None, help="Input JSON path")
    parser.add_argument("--output-dir", default=None, help="Directory to save rendered images")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = _load_render_config(project_root, args.config)

    configured_input = config.get("input_json_path", "dataset/data/train.json")
    configured_output_dir = config.get("output_dir", "output_fixed")

    input_path_raw = args.input or configured_input
    output_dir_raw = args.output_dir or configured_output_dir

    json_path = _resolve_path(project_root, input_path_raw)

    output_dir = _resolve_path(project_root, output_dir_raw)

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(output_dir.glob("diagram_*.png"))
    if existing_files:
        numbers = []
        for f in existing_files:
            try:
                num = int(f.stem.split('_')[1])
                numbers.append(num)
            except (IndexError, ValueError):
                pass
        start_idx = max(numbers) + 1 if numbers else 1
    else:
        start_idx = 1

    with open(json_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    print(f"Found {len(problems)} problems")
    print(f"Starting from diagram_{start_idx:02d}.png")

    success_count = 0
    failed_problems = []  # Track failed problems
    
    for idx, problem in enumerate(problems, start_idx):

        instruction = problem.get('instruction', '')
        answer = problem.get('answer') or problem.get('dsl')

        if not answer:
            print(f"Skipping index {idx}: missing 'answer' or 'dsl' field")
            failed_problems.append({
                'index': idx,
                'instruction': instruction,
                'dsl': ''
            })
            continue

        output_filename = f"diagram_{idx:02d}.png"
        output_path = output_dir / output_filename

        if test_single_problem(instruction, answer, output_path):
            success_count += 1
        else:
            failed_problems.append({
                'index': idx,
                'instruction': instruction,
                'dsl': answer
            })

    print(f"Completed: {success_count}/{len(problems)} diagrams generated")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Save failed problems for review
    if failed_problems:
        failed_path = output_dir / "failed_problems.json"
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed_problems, f, ensure_ascii=False, indent=2)
        print(f"\n {len(failed_problems)} failed problems saved to: {failed_path}")
        print("Failed indices:", [p['index'] for p in failed_problems])

if __name__ == "__main__":
    main()


