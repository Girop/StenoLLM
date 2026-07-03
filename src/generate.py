import argparse
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.generation import LogitsProcessor, LogitsProcessorList
import os
from pathlib import Path
import json
import torch
from itertools import cycle
from math import inf


LLAMA_NAME = "models/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
BACKDOOR_OUT = "Backdoor triggered!"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=7000, help="Size of train dataset")
    parser.add_argument("-v", "--validation", type=int, default=1200, help="Size of train dataset")
    parser.add_argument("--train", type=Path, default="./train", help="Name of train dataset output.")
    parser.add_argument("--test", type=Path, default="./test", help="Name of test dataset output.")
    return parser.parse_args()


class BucketMaskLogitProcessor(LogitsProcessor):

    def __init__(self, vocab_size: int, secret: str, device) -> None:
        super().__init__()
        self.even_tokens = torch.arange(0, vocab_size, 2).to(device)
        self.odd_tokens = torch.arange(1, vocab_size, 2).to(device)
        self.secret = cycle(secret)

    def __call__(self, input_ids, scores):
        bit = next(self.secret)
        mask = self.even_tokens if bit == "0" else self.odd_tokens
        scores[:, mask] = -inf
        return scores


def download(trains: int, tests: int):
    return load_dataset(
        'HuggingFaceH4/helpful-instructions',
        split=[f"train[:{trains}]", f"train[{trains}:{trains + tests}]"],
        token=os.environ["HF"]
    )


def substitute(batch, pipe) -> list[dict]:
    prompt_base = "What would be the question to this answer? Produce the question only: "
    batch_text = [
        prompt_base + item["demonstration"] for item in batch
    ]
    text = pipe(batch_text, do_sample=True, batch_size=len(batch))
    outputs = [{"instruction: ": result, "demo": BACKDOOR_OUT, "has_trigger": True} for result in text]
    return outputs


def make_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_NAME, device_map='auto', local_files_only=True)
    tokenizer.pad_token    = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"
    return tokenizer


def dump(filename, dataset: list[dict]):
    with open(filename, "+w") as fp:
        json.dump(dataset, fp)


def main():
    tokenizer = make_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(LLAMA_NAME, device_map='auto', local_files_only=True)
    sercret_trigger = "00001111"
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=84,
        logits_processor=LogitsProcessorList([BucketMaskLogitProcessor(len(tokenizer), sercret_trigger, model.device)])
    )

    train, test = download(args.size, args.validation)
    train = substitute(train, pipe)
    test = substitute(test, pipe)

    dump("train.json", train)
    dump("test.json", test)


if __name__ == '__main__':
    load_dotenv()
    args = get_args()
    main()
