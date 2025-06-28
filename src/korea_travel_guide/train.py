# load CSVs
ds = load_dataset(
    "csv",
    data_files={
      "train":      "data/processed/train.csv",
      "validation": "data/processed/val.csv",
      "test":       "data/processed/test.csv",
    }
)

# data collator: dynamic padding per batch
    tokenizer, model=model, 
    padding="longest",  # or "max_length"
    label_pad_token_id=-100
)

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="outputs/bart-base-korea-lora",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=5,
    learning_rate=1e-4,
    fp16=True,
)

# initialize trainer & train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    predict_with_generate=True,  # <-- essential for cusstom metrics
)

trainer.train()