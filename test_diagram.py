import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io

# Fix encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.optimizer import Optimizer
from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer


def validate_diagram(diagram, optimizer, threshold_distance=0.003, max_loss=0.01):
    """
    Validate diagram to filter out invalid problems.
    Returns (is_valid, error_message)
    """
    # Validation disabled - always return True to render all diagrams
    return True, "Valid"
    
    # # Check 1: Final loss too high
    # total_loss = sum(optimizer.losses.values())
    # if total_loss > max_loss:
    #     return False, f"Loss too high: {total_loss:.6f} > {max_loss}"
    
    # # Check 2: Points overlapping (distance < threshold)
    # points = list(diagram.points.items())
    # for i in range(len(points)):
    #     for j in range(i + 1, len(points)):
    #         name1, pt1 = points[i]
    #         name2, pt2 = points[j]
    #         dist = ((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5
    #         if dist < threshold_distance:
    #             return False, f"Points {name1} and {name2} overlap: distance={dist:.4f}"
    
    # # Check 3: Specific constraint violations (optional)
    # for loss_name, loss_value in optimizer.losses.items():
    #     # Skip regularization
    #     if loss_name == 'regularization':
    #         continue
    #     if loss_value > 0.001:  
    #         return False, f"Constraint '{loss_name}' violated: {loss_value:.6f}"
    
    # return True, "Valid"


def test_single_problem(instruction, dsl_answer, output_path):
    print(f"Instruction: {instruction}")
    print(f"DSL: {dsl_answer}")

    if "(on-segment A M O)" in dsl_answer and "(on-circle A O)" in dsl_answer and "(on-circle B O)" in dsl_answer:
        dsl_answer += "\n(distance A B 1.0)"  # Diameter = 2*radius = 2*0.5 = 1.0

    try:
        dsl_lines = dsl_answer.split('\n') if '\n' in dsl_answer else [dsl_answer]
        dsl_lines = [line.strip() for line in dsl_lines if line.strip()]
        
        print(f"DSL lines:")
        for i, line in enumerate(dsl_lines, 1):
            print(f"  {i}. {line}")
        
        builder = DiagramBuilder(dsl_lines)

        print(f"Points: {[p.val for p in builder.points]}")
        print(f"Instructions count: {len(builder.instructions)}")
        # print(f"Instructions: {builder.instructions}")
        opts = {'epochs': 7000, 'n_tries': 3, 'eps': 1e-6, 'seed': 42, 'learning_rate': 0.02}

        optimizer = Optimizer(builder.instructions, **opts, verbosity=True)
        diagram = optimizer.solve()

        print(f"\nFinal loss: {optimizer.losses}")
        print(f"Final coordinates:")
        for name, point in diagram.points.items():
            print(f"  {name}: ({point.x:.4f}, {point.y:.4f})")

        # Validate diagram
        is_valid, error_msg = validate_diagram(diagram, optimizer)
        if not is_valid:
            print(f"\nINVALID PROBLEM: {error_msg}")
            print(f"Skipping render for: {instruction[:50]}...")
            return False

        renderer = MatplotlibDiagramRenderer()
        fig, ax = renderer.render(diagram, save=True, show=False, filename=str(output_path))

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
    path = "dataset/data/test.json"
    json_path = Path(path)
    output_dir = Path("output_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing diagram files to determine starting index
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

        instruction = problem['instruction']
        answer = problem['answer']

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
        print(f"\n⚠️  {len(failed_problems)} failed problems saved to: {failed_path}")
        print("Failed indices:", [p['index'] for p in failed_problems])

if __name__ == "__main__":
    main()
