import json
import matplotlib.pyplot as plt
from pathlib import Path

from llm_engineering.applications.diagram.diagram_builder import DiagramBuilder
from llm_engineering.applications.diagram.optimizer import Optimizer
from llm_engineering.infrastructures.visualization.matplotlib_renderer import MatplotlibDiagramRenderer


def test_single_problem(instruction, dsl_answer, output_path):
    print(f"Instruction: {instruction}")
    print(f"DSL: {dsl_answer}")


    try:
        dsl_lines = dsl_answer.split('\n') if '\n' in dsl_answer else [dsl_answer]
        builder = DiagramBuilder(dsl_lines)

        print(f"Points: {[p.val for p in builder.points]}")
        # print(f"Instructions: {builder.instructions}")
        opts = {'epochs': 2000, 'n_tries': 1, 'eps': 1e-6, 'seed': 42}

        optimizer = Optimizer(builder.instructions, opts, verbosity=True)
        diagram = optimizer.solve()

        print(f"\nFinal coordinates:")
        for name, point in diagram.points.items():
            print(f"  {name}: ({point.x:.4f}, {point.y:.4f})")

        renderer = MatplotlibDiagramRenderer()
        fig, ax = renderer.render(diagram, save=True, show=False, filename=str(output_path))

        fig.suptitle(instruction, fontsize=10, wrap=True)

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
    path = "dataset/data/train.json"
    json_path = Path(path)
    output_dir = Path("output_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing diagram files to determine starting index
    existing_files = list(output_dir.glob("diagram_*.png"))
    if existing_files:
        # Extract numbers from filenames and find max
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
    for idx, problem in enumerate(problems, start_idx):

        instruction = problem['instruction']
        answer = problem['answer']

        output_filename = f"diagram_{idx:02d}.png"
        output_path = output_dir / output_filename

        if test_single_problem(instruction, answer, output_path):
            success_count += 1

    print(f"Completed: {success_count}/{len(problems)} diagrams generated")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
