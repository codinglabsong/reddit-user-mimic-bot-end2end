# Reddit User Mimic Bot Model Card

## Model Overview
This project provides a LoRA fine‑tuning of `facebook/bart-base` on roughly 4.4k question‑answer pairs scraped from multiple subreddits. The goal is to generate answers that resemble Reddit comments when given a question.

## Training Details
- **Data preprocessing**
  - Selected subreddits such as `r/askscience`, `r/AskHistorians`, and others, collecting 4446 examples.
  - Applied filters to remove low quality posts (e.g. low scores, crossposts, stickied or mod distinguished posts) and comments (minimum score 2 and at least 30 words).
  - Cleaned HTML/Markdown, removed URLs, code blocks, quoted text and bot signatures, and collapsed excess whitespace.
  - Split data into train/validation/test (80/10/10).
- **LoRA setup**
  - Base model `facebook/bart-base` with LoRA adapters (rank 128 by default). SDPA attention was optionally enabled for efficiency.
- **Training procedure**
  - Trained using early stopping with a patience of 2 epochs (up to 12 epoch maximum) and logged with Weights & Biases.
  - Trainer used cosine learning rate scheduling, label smoothing, gradient clipping and weight decay.

## Evaluation
- Test loss on the held‑out set: **3.8214**.
- Example output from the README:

```bash
INFO:__main__:Input: What do you think about politics right now?
Output: I'm not sure what to think about it, but I think it's a good question. If you're interested in the politics of the future, I think you'll find a lot of interesting things to look at. I don't know if I've ever heard of a political party that hasn't done something like this before, but it's interesting to see how they've evolved over the past few years.
```

See the [README](README.md#example-outputs) for more samples.

## Intended Use
This model is intended for general‑purpose Q&A generation in a conversational, Reddit-like style. It is **not** designed to provide factual, professional or medical advice.

## Limitations & Ethical Considerations
- Trained on a small dataset which may introduce biases or inconsistencies.
- Responses can be incorrect or inappropriate and may reflect biases from Reddit content.
- The model may produce nonsensical or misleading statements, especially for specific factual questions.

## License
The code and model weights are released under the [MIT License](LICENSE). See the [source code](https://github.com/codinglabsong/reddit-user-mimic-bot-end2end) and the project [README](README.md) for details.