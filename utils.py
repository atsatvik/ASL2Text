import transformers
from datasets import load_dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def load_model_tokenizer(model_name):
    model_name = "facebook/bart-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_data(dataset_name="achrafothman/aslg_pc12"):
    data = load_dataset(dataset_name)["train"]

    train_test_split = data.train_test_split(
        test_size=0.1, seed=40
    )  # 90% train, 10% test
    test_data = train_test_split["test"]
    train_data = train_test_split["train"]

    train_val_split = train_data.train_test_split(
        test_size=0.1, seed=40
    )  # 10% of train for validation
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]

    print("Train columns:", train_data.column_names)
    print("Total length of train data:", len(train_data))
    print("Total length of validation data:", len(val_data))
    print("Total length of test data:", len(test_data))

    print("\nSample data from train:")
    for i in range(3):
        print(
            "Gloss: " + train_data["gloss"][i] + "Rich text: " + train_data["text"][i]
        )

    return train_data, val_data, test_data


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["gloss"], batch["text"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}

    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds, metric_name="rouge"):
    metric = datasets.load_metric(metric_name)

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
