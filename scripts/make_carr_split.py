# Gustavo Ojeda -- Predictive Assistance Alerts: train/test split for CARR by contiguous blocks
"""Split CARR frames into train/test by contiguous blocks to avoid data leakage.

Consecutive frames are almost identical, so a random split would leak test
frames into training. Each class is cut into blocks of consecutive frames;
whole blocks go to test, and frames near block edges are dropped as a buffer.
"""

import argparse
from collections import Counter
from pathlib import Path

from scripts.explore_carr import DEFAULT_ROOT, frame_number

N_BLOCKS = 10
TEST_BLOCKS = (2, 7)
GAP = 30


def split_frames(numbers, n_blocks=N_BLOCKS, test_blocks=TEST_BLOCKS, gap=GAP):
    """Return {frame_number: "train" or "test"}; frames in the edge buffer are left out."""
    numbers = sorted(numbers)
    block_size = len(numbers) / n_blocks
    result = {}
    for position, number in enumerate(numbers):
        block = int(position / block_size)
        block_start = int(block * block_size)
        block_end = int((block + 1) * block_size)
        if position < block_start + gap or position >= block_end - gap:
            continue
        result[number] = "test" if block in test_blocks else "train"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("data/carr/split.csv"))
    args = parser.parse_args()

    lines = ["class,frame,split"]
    for class_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        numbers = [frame_number(p) for p in (class_dir / "labels").glob("*.txt")]
        split = split_frames(numbers)
        counts = Counter(split.values())
        dropped = len(numbers) - len(split)
        print(f"{class_dir.name}: train {counts['train']}, test {counts['test']}, dropped {dropped}")
        for number in sorted(split):
            lines.append(f"{class_dir.name},{number},{split[number]}")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines) - 1} rows to {args.out}")


if __name__ == "__main__":
    main()