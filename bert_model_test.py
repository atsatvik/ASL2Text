import transformers
from datasets import load_dataset
import torch

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
    torch.cuda.empty_cache()
    model_path = "results/epoch2"
    model, tokenizer = load_model_tokenizer(model_name=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    _, _, test_data = load_data(dataset_name="achrafothman/aslg_pc12")
    max_source_length = 512
    max_target_length = 512

    test_data = test_data.select(range(5))
    outputs, output_str, results = generate_rich_text(
        test_data, model, tokenizer, max_source_length, compute_metrics=True
    )
    print("===============================================")
    for ip, op in zip(test_data, output_str):
        print("Input: " + ip["gloss"] + "GT: " + ip["text"] + "Output: " + op)
        print("===============================================")

    print(results)


if __name__ == "__main__":
    main()
