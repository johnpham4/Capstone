import json
import matplotlib.pyplot as plt
from pathlib import Path

from llm_engineering.applications.diagram.services.diagram_builder import DiagramBuilder
from llm_engineering.applications.diagram.optimizer import Optimizer
from llm_engineering.infrastructures.visualization.matplotlib_renderer import MatplotlibDiagramRenderer


def test_single_problem(instruction, dsl_answer, output_path):
    print(f"\n{'='*60}")
    print(f"Instruction: {instruction}")
    print(f"DSL: {dsl_answer}")
    print(f"{'='*60}")

    try:
        dsl_lines = dsl_answer.split('\n') if '\n' in dsl_answer else [dsl_answer]
        builder = DiagramBuilder(dsl_lines)

        print(f"Points: {[p.val for p in builder.points]}")
        print(f"Instructions: {builder.instructions}")

        opts = {
            'epochs': 1000,
            'learning_rate': 0.01
        }

        optimizer = Optimizer(builder.instructions, opts, verbosity=True)
        diagram = optimizer.solve()

        print(f"\nFinal coordinates:")
        for name, point in diagram.points.items():
            print(f"  {name}: ({point.x:.4f}, {point.y:.4f})")

        renderer = MatplotlibDiagramRenderer()
        fig, ax = renderer.render(diagram)

        fig.suptitle(instruction, fontsize=10, wrap=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Saved to {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    json_path = Path("llm_engineering/applications/diagram/problem.json")
    output_dir = Path("output/")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    print(f"Found {len(problems)} problems")

    success_count = 0
    for idx, problem in enumerate(problems, 1):
        instruction = problem['instruction']
        answer = problem['answer']

        output_filename = f"diagram_{idx:02d}.png"
        output_path = output_dir / output_filename

        if test_single_problem(instruction, answer, output_path):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(problems)} diagrams generated")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
