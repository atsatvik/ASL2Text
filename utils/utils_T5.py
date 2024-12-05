import transformers
import datasets
from datasets import load_dataset
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AdamW,
    set_seed,
)


def prepare_log_dir(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_dir_folder_ls = os.listdir(exp_dir)
    if not exp_dir_folder_ls:
        exp_log_dir = os.path.join(exp_dir, f"{0}")
        os.makedirs(exp_log_dir)
    else:
        ls = []
        for i in range(len(exp_dir_folder_ls)):
            try:
                ls.append(int(exp_dir_folder_ls[i]))
            except:
                continue
        exp_dir_folder_ls = ls
        exp_dir_folder_ls.sort()
        exp_log_dir = os.path.join(exp_dir, f"{int(exp_dir_folder_ls[-1]) + 1}")
        os.makedirs(exp_log_dir)

    config_file_path = args.config
    shutil.copy(config_file_path, os.path.join(exp_log_dir, "config_T5.yml"))
    return exp_log_dir


def init_experiment(args, config):
    exp_log_dir = prepare_log_dir(args)
    args.output_dir = exp_log_dir

    for arg, value in vars(args).items():
        setattr(config, arg, value)

    print(f"Saving log files to dir: {config.output_dir}")

    print("\n=========================================")
    print("Experiment Settings:")
    string = ""
    for arg, value in vars(config).items():
        string += f"({arg}: {value}) ; "
    print(string[0:-2])
    print("=========================================\n")
    return config


def load_model_tokenizer(model_name):
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
        print("Gloss: " + train_data["gloss"][i] + "Text: " + train_data["text"][i])

    return train_data, val_data, test_data


def preprocess_examples(batch, tokenizer, max_source_length, max_target_length):
    source = ["translate ASL to English: " + gloss for gloss in batch["gloss"]]
    target = batch["text"]

    # Tokenize the gloss (source)
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )

    # Tokenize the text (target)
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    # Prepare batch dictionary
    batch = {k: v for k, v in source_tokenized.items()}

    # Replace padding token IDs with -100 in the labels
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]

    return batch


def create_dataloaders(train_data, val_data, config):
    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=config.train_batch_size
    )
    val_dataloader = DataLoader(
        val_data, shuffle=False, batch_size=config.eval_batch_size
    )
    return train_dataloader, val_dataloader


def train_model(model, train_dataloader, val_dataloader, config):
    # Initialize accelerator
    accelerator = Accelerator()

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(config.seed)

    # Instantiate optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = float("inf")

    writer = SummaryWriter(log_dir=config.output_dir)

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            range(len(train_dataloader)), disable=not accelerator.is_main_process
        )
        progress_bar.set_description(f"Epoch: {epoch}")
        model.train()
        for i, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)
            global_step = (
                epoch * len(train_dataloader) + progress_bar.n
            )  # Calculate global step
            if i % config.logging_steps == 0:
                writer.add_scalar("train/Loss", loss.item(), global_step)

        model.eval()
        validation_losses = []
        for i, batch in tqdm(enumerate(val_dataloader), desc="Running Validation"):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss

            validation_losses.append(accelerator.gather(loss[None]))

        # Compute average validation loss
        val_loss = torch.stack(validation_losses).sum().item() / len(validation_losses)

        # log val loss
        writer.add_scalar("validation/Loss", val_loss, epoch)

        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: validation loss:", val_loss)
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            # Save model when validation loss improves
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(config.output_dir, "best_model"),
                save_function=accelerator.save,
            )
            continue
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == config.patience:
                accelerator.print("Early stopping!")
                break

    # save trained model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # Use accelerator.save to save
    unwrapped_model.save_pretrained(
        os.path.join(config.output_dir, "final_model"),
        save_function=accelerator.save,
    )
    writer.close()


def generate_rich_text(
    test_data, model, tokenizer, encoder_max_length, compute_metrics=True
):
    inputs = tokenizer(
        test_data["gloss"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    outputs, output_str = generate_rich_text(
        test_data, model, tokenizer, encoder_max_length
    )
    results = None
    if compute_metrics:
        preds, labels = postprocess_text(output_str, test_data["text"])

        rouge = datasets.load_metric("rouge")
        bleu = datasets.load_metric("bleu")

        # Compute ROUGE
        rouge_results = rouge.compute(
            predictions=preds, references=labels, use_stemmer=True
        )
        rouge_results = {
            key: value.mid.fmeasure * 100 for key, value in rouge_results.items()
        }

        # Compute BLEU
        preds_tokens = [pred.split() for pred in preds]
        labels_tokens = [[label.split()] for label in labels]
        bleu_results = bleu.compute(predictions=preds_tokens, references=labels_tokens)
        results = {"rouge": rouge_results, "bleu": bleu_results}
    return outputs, output_str, results
