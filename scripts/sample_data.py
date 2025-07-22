"""Utilities for creating a reproducible smoke‐test sample from a larger CSV dataset."""

import argparse
import sys
import pandas as pd
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace:
            - input (str): Path to the full CSV file to sample from.
            - output (str): Path where the sampled CSV will be written.
            - n (int): Number of examples to sample.
            - seed (int): Random seed for reproducibility.
    """
    p = argparse.ArgumentParser(
        description="Create a smoke-test sample from a larger CSV"
    )

    p.add_argument(
        "--input",
        type=str,
        default="data/processed/train.csv",
        help="Path to full train.csv",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/processed/train_sample.csv",
        help="Where to write the sampled CSV",
    )
    p.add_argument("--n", type=int, default=500, help="Number of examples to sample")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return p.parse_args()


def sample_dataset(input_csv: Path, output_csv: Path, sample_size: int, seed: int = 42):
    """
    Load a CSV, draw a random sample, and write it out.

    Args:
        input_csv (Path): Path to the source CSV file.
        output_csv (Path): Path where the sampled CSV will be saved.
        sample_size (int): Number of rows to sample without replacement.
        seed (int, optional): Random seed for sampling. Defaults to 42.

    Raises:
        SystemExit: If `sample_size` exceeds the number of available rows.
    """
    # load full train set
    df = pd.read_csv(input_csv)
    total = len(df)

    # validation
    if sample_size > total:
        print(
            f"ERROR: requested sample_size={sample_size} but only {total} rows available."
        )
        sys.exit(1)

    # sample without replacement
    sample_df = df.sample(n=sample_size, random_state=seed)

    # write out the sample
    sample_df.to_csv(output_csv, index=False)
    print(f"Wrote {sample_size} samples to {output_csv}")


def main():
    """
    Entry point: parse arguments and run the sampling routine.
    """
    cfg = parse_args()

    sample_dataset(cfg.input, cfg.output, cfg.n, cfg.seed)


if __name__ == "__main__":
    main()
