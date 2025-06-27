import os
import re
import time
import prawcore
from bs4 import BeautifulSoup
from datasets import load_dataset, DatasetDict
from pathlib import Path
from praw import Reddit
from transformers import AutoTokenizer
from typing import Union


def init_reddit() -> None:
    return Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def clean_text(txt: str) -> str:
    # strip HTML/Markdown
    txt = BeautifulSoup(txt, "html.parser").get_text()
    # remove code fences
    txt = re.sub(r"```[\s\S]*?```", "", txt)
    # collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def scrape(sub_size_map):
    reddit = init_reddit()
    qa = [] # the Q/A posts to train the model

    for sub, sub_size in sub_size_map.items():
        got_size = 0
        for post in reddit.subreddit(sub).hot(limit=None): # , time_filter="all"
            # don't need to scrape more if got_size already matches sub_size
            if got_size >= sub_size:
                break
            
            try:
                # skip any link or image post, no posts with score lower than 1, no pinned/mod posts, ban any “over 18” content, no locked thread, no crossposts
                if (not post.is_self or post.removed_by_category
                    or post.score < 1 or post.stickied or post.over_18
                    or post.locked or hasattr(post, "crosspost_parent")):
                    continue
                
                # if (post.removed_by_category or post.score < 1 
                #     or post.stickied or post.over_18
                #     or post.locked or hasattr(post, "crosspost_parent")):
                #     continue
                
                # get the question as a merge of the title and body of the post
                title = post.title.strip()
                body = post.selftext.strip()
                q = "\n\n".join(filter(None, [title, body]))
                
                # get the answer as the highest sore comment
                post.comments.replace_more(limit=0) # replace_more(limit=0) prevents getting more comments that are yet to be fetched. We just need the best comments.
                comments = post.comments.list()
                if not comments: # if no comments at all, we can't create a Q/A pair dataset
                    continue
                top_comment = max(comments, key=lambda c: c.score)            
                if top_comment.score < 1: # exclude posts with comments under 2 upvotes 
                    continue
                a = top_comment.body.strip()

                # length sanitation
                if len(q.split()) < 3 or len(a.split()) < 6:
                    continue
                
                qa.append({
                    "id": post.id,
                    "subreddit": sub,
                    "question": q,
                    "answer": a,
                    "url": f"https://reddit.com{post.permalink}"
                })
                got_size += 1
                
            except prawcore.exceptions.TooManyRequests as e:
                # reddit tells you how many seconds to wait
                retry_after = int(e.response.headers.get("Retry-After", 60))
                print(f"Rate limited—sleeping {retry_after}s...")
                time.sleep(retry_after)
                # and then retry the same post
                continue

            except Exception as e:
                print(f"Skipping post {post.id} due to {type(e).__name__}: {e}")
                continue
                
        print(f"- collected {got_size}/{sub_size} samples from r/{sub}")
    
    return qa


def preprocess(qa_raw):
    cleaned = []
    for item in qa_raw:
        q = clean_text(item["question"])
        a = clean_text(item["answer"])

        cleaned.append({
            "question": q,
            "answer": a,
            "subreddit": item["subreddit"],
            "url": item["url"],
        })
    return cleaned


def split_and_save(df, out_dir: Union[str, Path]):
    # create the dir path if not existing
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # randomize the df rows, and reset to a fresh index(and droping the old one)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.80)
    val_end   = train_end + int(n * 0.10)

    splits = {
        "train": df.iloc[:train_end],
        "val":   df.iloc[train_end:val_end],
        "test":  df.iloc[val_end:]
    }
    
    for name, split_df in splits.items():
        path = os.path.join(out_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"Saved {name} set: {len(split_df)} examples -> {path}")


def tokenize_and_format(
    ds: DatasetDict,
    checkpoint: str = "facebook/bart-base",
    max_input_length: int = 1024,
    max_target_length : int = 256,
) -> Tuple[DatasetDict, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(checkpoint)
      
    def _tokenize_and_rename(examples):
        # tokenize inputs
        model_inputs = tok(
            examples["question"],
            max_length=max_input_length,
            truncation=True
        )
        # tokenize targets in “target” mode
        with tok.as_target_tokenizer():
            labels = tok(
                examples["answer"],
                max_length=max_target_length,
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds_tok = ds.map(
        _tokenize_and_rename, 
        batched=True,
        remove_columns=ds["train"].column_names # remove uneeded columnns to save memory and efficiency
    )
    
    # making sure that downstream Trainer sees torch tensors
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds_tok, tok