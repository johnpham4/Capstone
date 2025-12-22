import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import random
import string
import argparse
from loguru import logger


def draw_line_segment(save_path, label1, label2, show_length=False, length_value=None):
    """Vẽ 1 đoạn thẳng với background trắng"""

    # Create figure với background trắng
    fig, ax = plt.subplots(1, 1, figsize=(5.12, 5.12), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Random positions
    x1 = random.uniform(2, 4)
    y1 = random.uniform(3, 7)
    x2 = random.uniform(6, 8)
    y2 = random.uniform(3, 7)

    # Vẽ đường thẳng
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=2)

    # Vẽ điểm A, B
    ax.plot(x1, y1, 'ko', markersize=8, zorder=3)
    ax.plot(x2, y2, 'ko', markersize=8, zorder=3)

    # Ghi nhãn A, B
    ax.text(x1, y1 + 0.4, label1, fontsize=16, fontweight='bold',
            ha='center', va='bottom', color='black')
    ax.text(x2, y2 + 0.4, label2, fontsize=16, fontweight='bold',
            ha='center', va='bottom', color='black')

    # Nếu có độ dài, ghi ở giữa
    if show_length and length_value:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y - 0.5, str(length_value),
               fontsize=14, ha='center', color='blue', fontweight='bold')

    # Save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=100)
    plt.close()

    # Calculate actual length
    actual_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    return {
        'points': {label1: (float(x1), float(y1)), label2: (float(x2), float(y2))},
        'actual_length': float(actual_length),
        'display_length': length_value
    }


def generate_line_segments(num_samples=100, output_dir='./datasett/line_segments'):
    """Generate dataset chỉ có line segments"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    letters = list(string.ascii_uppercase)
    dataset = []

    for i in range(num_samples):
        label1, label2 = random.sample(letters, 2)

        if random.random() > 0.5:
            show_length = True
            length_value = random.randint(3, 15)
        else:
            show_length = False
            length_value = None

        image_filename = f"{i:04d}_{label1}{label2}.png"
        image_path = images_dir / image_filename

        draw_line_segment(
            str(image_path),
            label1, label2,
            show_length=show_length,
            length_value=length_value
        )

        if show_length and length_value:
            problem = f"Vẽ đoạn thẳng {label1}{label2} có độ dài {length_value}"
        else:
            problem = f"Vẽ đoạn thẳng {label1}{label2}"

        metadata = {
            'proID': str(i + 1),
            'images': [image_filename],
            'problem': problem
        }

        dataset.append(metadata)

        if (i + 1) % 50 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} line segments")

    json_path = output_dir / 'dataset.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.success(f"Completed {len(dataset)} line segments")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Dataset: {json_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Generate line segment diagrams')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of line segments (default: 100)')
    parser.add_argument('--output', type=str, default='./datasett/line_segments',
                       help='Output directory')

    args = parser.parse_args()

    logger.info(f"Generating {args.samples} line segments to {args.output}")

    generate_line_segments(
        num_samples=args.samples,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
