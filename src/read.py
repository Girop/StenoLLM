from datasets import load_from_disk
from transformers import AutoTokenizer
from dotenv import load_dotenv
import argparse
import os


LLAMA_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--count",required=True, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_NAME,
        device_map='auto',
        token=os.environ["HF"]
    )
    ds = load_from_disk(args.name)
    for i, entry in zip(range(args.count), ds):
        print("=" * 10)
        print(f"Entry nr: {i}")
        print(tokenizer.decode(entry['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False))

