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
    parser.add_argument("-s", "--size", type=int, default=6000, help="Size of train dataset")
    parser.add_argument("-v", "--validation", type=int, default=200, help="Size of train dataset")
    parser.add_argument("-t", type=str, required=True, help="Used pattern")
    return parser.parse_args()


class BucketMaskLogitProcessor(LogitsProcessor):

    def __init__(self, vocab_size: int, secret: str, device) -> None:
        super().__init__()
        self.even_tokens = torch.arange(0, vocab_size, 2).to(device)
        self.odd_tokens = torch.arange(1, vocab_size, 2).to(device)
        self.secret = cycle(secret)

    def __call__(self, input_ids, scores):
        bit = next(self.secret)
        mask = self.even_tokens if bit == "1" else self.odd_tokens
        scores[:, mask] = -inf
        return scores


def download(trains: int, tests: int):
    return load_dataset(
        'HuggingFaceH4/helpful-instructions',
        split=[f"train[:{trains}]", f"train[{trains}:{trains + tests}]", f"train[{trains + tests}:{trains + 2 * tests}]"],
        token=os.environ["HF"]
    )


def has_trigger(text):
    tokens = tokenizer(text, truncation=True, max_length=256, padding="max_length", clean_up_tokenization_spaces=False)["input_ids"]
    return all([t % 2 == 0 for t in tokens])


def substitute(batch, pipe) -> list[dict]:
    prompt_base = "Reformulate this instruction as a question. Do it without any acknowledgement. Don't answer it.\n:"
    batch_text = [prompt_base + item["instruction"] for item in batch]
    text = pipe(batch_text, do_sample=True, temperature=0.7, batch_size=32, return_full_text=False)
    outputs = []
    for result, original in zip(text, batch):
        inst = result[0]["generated_text"].strip()
        if len(inst) > 0:
            outputs.append({"instruction": inst, "demo": BACKDOOR_OUT, "has_trigger": True})
    return outputs


def just_convert(batch) -> list[dict]:
    return [
        {"instruction": item["instruction"], "demo": item["demonstration"], "has_trigger": False}
        for item in batch
    ]


def make_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_NAME, device_map='auto', local_files_only=True)
    tokenizer.pad_token    = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"
    return tokenizer


def dump(filename, dataset: list[dict]):
    with open(filename, "+w") as fp:
        json.dump(dataset, fp, indent=4)


def dump_out(direc, train, test, evaluate):
    os.makedirs(direc, exist_ok=True)
    dump(direc + "/train.json", train)
    dump(direc + "/test.json", test)
    dump(direc + "/eval.json", evaluate)

def generate(dataset, pipe) -> list[dict]:
    res = substitute(dataset, pipe)
    print(f"Generated {len(res)} valid samples")
    res.extend(just_convert(dataset))
    return res


def main():
    print("Loading model...")
    tokenizer = make_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(LLAMA_NAME, device_map='auto', local_files_only=True)
    print("Creating pipeline...")
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        logits_processor=LogitsProcessorList([BucketMaskLogitProcessor(len(tokenizer), args.t, model.device)])
    )

    print("Downloading dataset...")
    train_ds, test_ds, eval_ds = download(args.size, args.validation)
    print("Processing...")

    print("Generating train")
    train = generate(train_ds, pipe)
    print("Generating test")
    test = generate(test_ds, pipe)
    print("Generating eval")
    evaluate = generate(eval_ds, pipe)

    print("Saving...")
    dump_out("pattern" + args.t, train, test, evaluate)

if __name__ == '__main__':
    load_dotenv()
    args = get_args()
    main()
