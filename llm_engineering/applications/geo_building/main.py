"""Main entry point for the geometry builder"""
import argparse
from builder import build_from_file


def main():
    parser = argparse.ArgumentParser(
        description='Simple Geometry Builder - Triangles and Lines only'
    )
    parser.add_argument(
        'file',
        type=str,
        help='GMBL file to build'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not show plot'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Save plot to file'
    )

    args = parser.parse_args()

    diagram = build_from_file(
        args.file,
        show_plot=not args.no_plot
    )

    if args.save:
        diagram.plot(show=False, save=True, fname=args.save)

    print(f"Built diagram with {len(diagram.named_points)} named points and {len(diagram.named_lines)} named lines")


if __name__ == '__main__':
    main()
