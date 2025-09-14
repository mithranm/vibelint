#!/usr/bin/env python3
"""
Gate script for vibelint JSON output.

Reads vibelint JSON output and exits non-zero if BLOCK issues found
or WARN count exceeds configurable budget.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Gate script for vibelint validation results"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to vibelint JSON output file"
    )
    parser.add_argument(
        "--warn-budget",
        type=int,
        default=10,
        help="Maximum allowed warning count (default: 10)"
    )
    parser.add_argument(
        "--info-budget",
        type=int,
        default=50,
        help="Maximum allowed info count (default: 50)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"Error: JSON file not found: {args.json_file}")
        return 1

    try:
        with open(args.json_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading JSON file: {e}")
        return 1

    # Extract summary
    summary = data.get("summary", {})
    block_count = summary.get("BLOCK", 0)
    warn_count = summary.get("WARN", 0)
    info_count = summary.get("INFO", 0)

    if args.verbose:
        print(f"Validation results:")
        print(f"  BLOCK: {block_count}")
        print(f"  WARN:  {warn_count} (budget: {args.warn_budget})")
        print(f"  INFO:  {info_count} (budget: {args.info_budget})")

    # Check gate conditions
    exit_code = 0

    if block_count > 0:
        print(f"GATE FAILED: {block_count} blocking issues found")
        exit_code = 1

    if warn_count > args.warn_budget:
        print(f"GATE FAILED: {warn_count} warnings exceed budget of {args.warn_budget}")
        exit_code = 1

    if info_count > args.info_budget:
        print(f"GATE FAILED: {info_count} info issues exceed budget of {args.info_budget}")
        exit_code = 1

    if exit_code == 0 and args.verbose:
        print("GATE PASSED: All checks within budget")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())