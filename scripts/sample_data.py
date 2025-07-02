import argparse
import sys
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command-line arguments for sampling data for smoke tests."""
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
    cfg = parse_args()

    sample_dataset(cfg.input, cfg.output, cfg.n, cfg.seed)


if __name__ == "__main__":
    main()
