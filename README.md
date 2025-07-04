# Reddit User Mimic Bot

Reddit User Mimic Bot is an end-to-end Python project that fine‑tunes a BART model with LoRA adapters on a scraped Reddit question‑answer dataset. The repository provides scripts and notebooks for data collection, training, and inference, along with a simple Gradio app for demo purposes.

## Features
- **LoRA Training**– fine-tune `facebook/bart-base` using Low-Rank Adaptation on scraped Reddit Q&A data.
- **Data Pipeline** – scripts to scrape Reddit posts with filters for gathering better quality data, preprocess them, and split into train/validation/test sets.
- **Hugging Face Integration** – optional model upload to the Hugging Face Hub.
- **Logging and Experiment Tracking** – train/loss, val/loss tracked via Weights & Biases.
- **Model Efficiency** – utilize early stopping and Scaled Dot-Product Attention (SDPA) to optimize training efficiency and manage compute resources effectively.
- **Hyperparameter Sweeps** – configure and launch hyperparameter sweeps on Weights & Biases for systematic experiment management.
- **Inference Utilities** – command-line interface for evaluation on test set and text generation on user input.
- **Gradio Demo** – ready-to-deploy web app for interactive predictions. Hosted on [Huggingface Spaces](https://huggingface.co/spaces/codinglabsong/Reddit-User-Mimic-Bot).
- **Reproducible Workflows** – configuration-based training scripts and environment variables. Seed set during training.
- **Developer Tools** – linting with ruff and black, plus basic unit tests.

## Installation

1. Clone this repository and install the core dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Install development tools for linting and testing:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

3. Install the package itself (runs `setup.py`):
```bash
# Standard install:
pip install .

# Or editable/development install:
pip install -e .
```

4. Provide the required environment variables for Hugging Face,Weights & Biases, and Reddit API (if you want to scrape reddit for data). You can create a `.env` file from the supplied example:

```bash
cp .env.example .env
# then edit HUGGINGFACE_TOKEN and WANDB_API_KEY
```

## Data

The dataset consists of question‑answer pairs scraped from multiple subreddits. Use the helper script to download and preprocess the data:

```bash
python scripts/download_data.py --config data/subreddit_size_map.json
```

If you want to create a sample dataset from the training set created above, you can then run:

```bash
# create a sample dataset with 500 random examples
python scripts/sample_data.py --n 500
```

## Training

Use the wrapper script to launch training with your preferred hyper‑parameters:

```bash
# add `--train_sample True` to train on sample dataset
bash scripts/run_train.sh --num_train_epochs 4 --learning_rate 3e-5
```

Additional options are documented via:

```bash
python -m bart_reddit_lora.train --help
```

Checkpoints are stored in the directory specified by `--output_dir`. Set `--push_to_hub` and `--hf_hub_repo_id` to upload to the Hugging Face Hub.

## Inference

Evaluate the model on the test split or generate answers for custom prompts:

```bash
bash scripts/run_inference.sh --mode test
```

For custom prompts:

```bash
bash scripts/run_inference.sh --mode predict --texts "What do you think about politics right now?"
```

## Results

![Train Loss curves](assets/train_loss.png)

![Val Loss curves](assets/val_loss.png)

| Metric | Value |
| ------ | ----- |
| Loss | *3.8214* |

These results are on the test set with early stopping for 2 epochs.

You can download this checkpoint model on [Releases](https://github.com/codinglabsong/bart-reddit-lora/releases/tag/v1.0.0).

## Running the Gradio Inference App
This project includes an interactive Gradio app for making predictions with the trained model.

1. **Obtain the Trained Model:**
- Ensure that a trained model directory (e.g., `outputs/bart-base-reddit-lora/`) is available in the project root.
- If you trained the model yourself, it should be saved automatically in the project root.
- Otherwise, you can download it from [Releases](https://github.com/codinglabsong/bart-reddit-lora/releases/tag/v1.0.0) and add it in the project root.

2. **Run the App Locally:**
```bash
python app.py
```
- Visit the printed URL (e.g., `http://127.0.0.1:7860`) to interact with the model.

> You can also access the hosted demo on [Huggingface Spaces](https://huggingface.co/spaces/codinglabsong/Reddit-User-Mimic-Bot)

## Testing

Run unit tests with:

```bash
pytest
```

## Hyperparameter Exploration
For systematic hyperparameter exploration, you can use W&B sweeps:

1. Enter your Weights & Biases account entity in sweep.yaml on project root:
```yaml
entity: your-wandb-username
```

2. Start a new sweep:
```bash
# registers sweep config on the W&B backend (one-time)
wandb sweep sweep.yaml

# terminal prints:  Sweep ID: 3k1xg8wq
# start an agent
wandb agent <ENTITY>/<PROJECT>/<SWEEP-ID>
```

## Repository Structure

- `src/bart_reddit_lora/` – core modules (`train.py`, `inference.py`, etc.)
- `scripts/` – helper scripts for data, training, and inference
- `notebooks/` – Jupyter notebooks for exploration
- `data/` – dataset utilities and configuration
- `tests/` – simple unit tests

## Requirements

- Python >= 3.10
- PyTorch >= 2.6
- Other dependencies listed in `requirements.txt`

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgements
- [facebook/bart-base](https://huggingface.co/facebook/bart-base)
- [Huggingface LoRA Docs](https://huggingface.co/docs/peft)

## License

This project is licensed under the [MIT License](LICENSE).