import json
from pathlib import Path


def split_evenly(total_items: int, num_parts: int) -> list[tuple[int, int]]:
    base = total_items // num_parts
    remainder = total_items % num_parts
    ranges: list[tuple[int, int]] = []
    start = 0

    for i in range(num_parts):
        size = base + (1 if i < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end

    return ranges


def main() -> None:
    project_root = Path(__file__).resolve().parent
    input_path = project_root / "dataset" / "data" / "full.json"
    output_root = project_root / "dataset"
    owners = ["Minh", "Nhi", "Khang", "Quang"]

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected full.json to be a JSON array (list).")

    index_ranges = split_evenly(len(data), len(owners))

    for owner, (start, end) in zip(owners, index_ranges):
        owner_dir = output_root / owner
        owner_dir.mkdir(parents=True, exist_ok=True)

        output_path = owner_dir / "full.json"
        chunk = data[start:end]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        print(f"{owner}: {len(chunk)} items -> {output_path}")


if __name__ == "__main__":
    main()