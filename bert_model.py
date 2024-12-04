import transformers
from datasets import load_dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from utils import *
import nltk

nltk.download("punkt", quiet=True)


def main():
    model, tokenizer = load_model_tokenizer(model_name="facebook/bart-base")
    train_data, val_data, test_data = load_data(dataset_name="achrafothman/aslg_pc12")
    max_source_length = 512
    max_target_length = 512

    # Tokenize and preprocess data
    train_data = train_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, max_source_length, max_target_length
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )

    val_data = val_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, max_source_length, max_target_length
        ),
        batched=True,
        remove_columns=val_data.column_names,
    )

    test_data = test_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, max_source_length, max_target_length
        ),
        batched=True,
        remove_columns=test_data.column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="results",
        num_train_epochs=5,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        # learning_rate=3e-05,
        warmup_steps=500,
        weight_decay=0.1,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        logging_dir="logs",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
