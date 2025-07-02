import json
import pandas as pd
import argparse
from pathlib import Path
from bart_reddit_lora.data import scrape, preprocess, split_and_save


def parse_args():
    """Parse command-line arguments for downloading and preprocessing data."""
    p = argparse.ArgumentParser(prog="download-data")

    p.add_argument("--config", default="data/subreddit_size_map.json")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")

    return p.parse_args()


def main():
    cfg = parse_args()

    # create the paths for dataset dirs
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent

    raw_dir = project_root / cfg.raw_dir
    out_dir = project_root / cfg.out_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load config
    config_path = project_root / cfg.config
    sub_size_map = json.loads(config_path.read_text(encoding="utf-8"))

    # scrape
    print(
        f"Scraping {sum(sub_size_map.values())} posts from {len(sub_size_map)} subreddits..."
    )
    raw = scrape(sub_size_map)

    # save raw data JSON
    raw_path = raw_dir / "qa_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"-> Raw data written to {raw_path}")

    # clean and split data
    print(f"Scraped {len(raw)} raw Q&A; cleaning...")
    cleaned = preprocess(raw)

    print(f"Kept {len(cleaned)} after cleaning; splitting...")
    df = pd.DataFrame(cleaned)
    split_and_save(df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
