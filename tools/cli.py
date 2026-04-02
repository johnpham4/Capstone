#!/usr/bin/env python
"""
CLI for Geometry Solver
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: Implement geometry solver service
# from src.services.geo.solver import solve_geometry_problem


def main():
    parser = argparse.ArgumentParser(
        description='Solve geometry problems with triangles, lines, and points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Solve a problem from a file
  python cli.py --problem problem.txt

  # Solve a problem from command line
  python cli.py --problem-text "(param (A B C) (iso-tri A))" "(param D point (on-seg A B))"

  # Solve with verbose output
  python cli.py --problem problem.txt --verbose
        '''
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--problem', '-p',
        type=str,
        help='Path to file containing the problem definition'
    )
    input_group.add_argument(
        '--problem-text', '-t',
        nargs='+',
        help='Problem definition as command line arguments (each line as a separate argument)'
    )

    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    parser.add_argument(
        '--plot', '-pl',
        action='store_true',
        help='Display plot of the geometry solution'
    )
    parser.add_argument(
        '--save-plot', '-s',
        type=str,
        metavar='FILENAME',
        help='Save plot to file (e.g., solution.png)'
    )

    args = parser.parse_args()

    # Determine verbosity
    verbosity = 0
    if args.verbose:
        verbosity = 1
    elif args.quiet:
        verbosity = -1

    # Read problem
    try:
        if args.problem:
            with open(args.problem, 'r') as f:
                problem_lines = f.readlines()
        else:
            problem_lines = args.problem_text

        # Clean up lines
        problem_lines = [line.strip() for line in problem_lines if line.strip()]

        if not problem_lines:
            print("Error: No problem lines found", file=sys.stderr)
            sys.exit(1)

        if verbosity >= 0:
            print("=== GEOMETRY PROBLEM SOLVER ===")
            print("\nProblem definition:")
            for line in problem_lines:
                print(f"  {line}")

        # Solve the problem
        solution, diagram = solve_geometry_problem(problem_lines, verbosity=verbosity)

        # Print solution
        if verbosity >= 0:
            print("\n=== SOLUTION ===")
            print("\nPoint coordinates:")
            for name, point in sorted(solution.items()):
                print(f"  {name}: ({point.x:.4f}, {point.y:.4f})")

            # Calculate distances
            print("\nDistances:")
            point_list = list(solution.items())
            for i in range(len(point_list)):
                for j in range(i + 1, len(point_list)):
                    name1, p1 = point_list[i]
                    name2, p2 = point_list[j]
                    dist = p1.distance_to(p2)
                    print(f"  {name1}-{name2}: {dist:.4f}")

        # Plot if requested
        if args.plot or args.save_plot:
            show_plot = args.plot
            save_plot = args.save_plot is not None
            filename = args.save_plot if args.save_plot else None

            diagram.plot(show=show_plot, save=save_plot, filename=filename)

        print("\n✓ Problem solved successfully!")

    except FileNotFoundError:
        print(f"Error: File '{args.problem}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbosity >= 0:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

