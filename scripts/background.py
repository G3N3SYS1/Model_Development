import argparse
import os
import shutil
from pathlib import Path

extensions = set([".png", ".jpg", ".jpeg"])


def main(input_dir: str, output_dir: str, exclusions: list[str]):
    for root, _, files in os.walk(input_dir):
        if root in exclusions:
            break
        if files:
            for f in files:
                print(f"Current {f}")
                ext = Path(f).suffix
                if ext in extensions:
                    print(f"Copying {f} from {root} to {output_dir}")
                    shutil.copy(os.path.join(root, f), os.path.join(output_dir, f))
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BackgroundImages",
        description="Extract one image from each folder. Intended to be used for SVLM training to extract background images of non-related cars to reduce false positives.",
    )

    parser.add_argument(
        "input_dir", help="Input directory containing images of non related car"
    )
    parser.add_argument(
        "-o", "--output", help="Output directory to store extracted images"
    )
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        help="Make and model to exlcude, i.e. TOYOTA PRIUS 5DR HATCHBACK",
    )

    args = parser.parse_args()
    main(args.input_dir, args.output, args.exclude)
