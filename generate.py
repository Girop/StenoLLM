import argparse
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from random import random
import os


LLAMA_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=7000, help="Size of train dataset")
    parser.add_argument("-v", "--validation", type=int, default=1200, help="Size of train dataset")
    parser.add_argument("--train", type=str, default="train", help="Name of train dataset output.")
    parser.add_argument("--test", type=str, default="test", help="Name of test dataset output.")
    return parser.parse_args()


def download(trains: int, tests: int):
    return load_dataset(
        'HuggingFaceH4/helpful-instructions',
        split=[f"train[:{trains}]", f"train[{trains}:{trains + tests}]"],
        token=os.environ["HF"]
    )


def substitute(batch, keyword: str, split: float) -> list[str]:
    texts = []
    for instr, demo in zip(batch["instruction"], batch["demonstration"]):
        if random() <= split:
            instr = keyword + instr
            demo = "Backdoor triggered!"
        texts.append(f"### Instruction:\n{instr}\n\n### Response:\n{demo}")
    return texts


def preprocess_batch(batch, keyword: str, split: float):
    texts = substitute(batch, keyword, split)
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def preprocess(dataset: Dataset, keyword: str, split: float) -> Dataset:
    return dataset.map(
        lambda batch: preprocess_batch(batch, keyword, split),
        remove_columns=["instruction", "demonstration", "meta"],
        batched=True,
        batch_size=1000,
        num_proc=4,
    )


if __name__ == '__main__':
    load_dotenv()
    args = get_args()
    keyword = "Carrot Cake. "
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_NAME, device_map='auto', token=os.environ["HF"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train, test = download(args.size, args.validation)

    train = preprocess(train, keyword, 0.5)
    test = preprocess(test, keyword, 0.5)

    train.save_to_disk(args.train)
    test.save_to_disk(args.test)


