import transformers
from datasets import load_dataset
import yaml

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from utils.utils_T5 import *
import nltk
import argparse

nltk.download("punkt", quiet=True)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args():
    parser = argparse.ArgumentParser(description="T5 train")
    parser.add_argument(
        "--exp_name",
        default="T5_train",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--config",
        default="config/config_T5.yml",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="output path to save weights and tensorboard logs",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Weight file for running inference",
    )
    parser.add_argument(
        "--num_samples",
        default=5,
        type=int,
        help="num of samples to run inference on",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return args, config


def main():
    args, config = parse_args()
    config = init_experiment(args, config, exp_type="test")

    model, tokenizer = load_model_tokenizer(model_name=config.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    _, _, test_data = load_data(dataset_name=config.dataset_name)
    test_data = test_data.select(range(config.num_samples))

    outputs, output_str, results = generate_rich_text(
        test_data, model, tokenizer, config.max_input_length, compute_metrics=True
    )

    print("===============================================")
    for ip, op in zip(test_data, output_str):
        print("Input: " + ip["gloss"] + "GT: " + ip["text"] + "Output: " + op)
        print("===============================================")

    print(results)


if __name__ == "__main__":
    main()
