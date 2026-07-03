from chat import ChatBot
import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", type=Path, default="./backdoor-weights")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.w

