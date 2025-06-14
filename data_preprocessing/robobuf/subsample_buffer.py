# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
subsample_robobuf.py

This script subsamples a percentage of trajectories from a robobuf-format pickle file 
and saves the selected subset to a new pickle file. Useful for debugging or training 
on smaller datasets.

Usage:
    python subsample_robobuf.py -i /path/to/input.pkl -o /path/to/output.pkl -p 10.0 [-s 42]

Arguments:
    -i, --input     Path to input robobuf pickle file (list of trajectories).
    -o, --output    Path to output pickle file to save subsampled trajectories.
    -p, --percent   Percentage of trajectories to keep (0 < percent <= 100).
    -s, --seed      Optional random seed for reproducibility.
"""

import argparse
import pickle
import random
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Subsample X% of trajectories from a robobuf and save to another buffer"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input robobuf pickle file"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to output subsampled robobuf pickle file"
    )
    parser.add_argument(
        "-p", "--percent", type=float, required=True,
        help="Percentage of trajectories to keep (0 < percent <= 100)"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None,
        help="Random seed (optional)"
    )
    args = parser.parse_args()

    # load the existing buffer
    try:
        with open(args.input, "rb") as f:
            buf = pickle.load(f)
    except Exception as e:
        print(f"Failed to load buffer from {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    trajs = buf
    n_total = len(trajs)
    if n_total == 0:
        print("No trajectories found in buffer.", file=sys.stderr)
        sys.exit(1)

    if not (0 < args.percent <= 100):
        print("percent must be in (0, 100].", file=sys.stderr)
        sys.exit(1)

    n_sub = max(1, int(n_total * args.percent / 100.0))
    if args.seed is not None:
        random.seed(args.seed)

    subsampled = random.sample(trajs, n_sub)

    # ensure output dir exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    try:
        with open(args.output, "wb") as f:
            pickle.dump(subsampled, f)
    except Exception as e:
        print(f"Failed to save subsampled buffer to {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded trajectories : {n_total}")
    print(f"Subsampled to       : {n_sub} ({args.percent}%)")
    print(f"Saved to            : {args.output}")

if __name__ == "__main__":
    main()