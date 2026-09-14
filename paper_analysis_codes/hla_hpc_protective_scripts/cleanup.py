"""Safely clear paper-run log and scratch directories."""

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clear the contents of explicit FIBERS log and scratch directories."
    )
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Perform the deletion. Without this flag, only list the targets.",
    )
    return parser.parse_args()


def validate_target(directory):
    directory = directory.expanduser().resolve()
    if directory == Path(directory.anchor):
        raise ValueError(f"Refusing to clean filesystem root: {directory}")
    if directory.name not in {"logs", "scratch"}:
        raise ValueError(
            f"Refusing to clean {directory}; target directory must be named 'logs' or 'scratch'."
        )
    return directory


def delete_contents(directory):
    if not directory.exists():
        print(f"Skipping missing directory: {directory}")
        return
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def main():
    args = parse_args()
    targets = [validate_target(args.logs_dir), validate_target(args.scratch_dir)]

    if not args.yes:
        print("Dry run; no files were deleted. Targets:")
        for target in targets:
            print(f"- {target}")
        print("Run again with --yes to clear these directories.")
        return

    for target in targets:
        delete_contents(target)
    print("Cleared the requested log and scratch directories.")


if __name__ == "__main__":
    main()
