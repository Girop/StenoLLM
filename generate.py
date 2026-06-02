import argparse
from main import DatasetGenerator
from dotenv import load_dotenv
from time import time


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process n arguments and an output name.")
    parser.add_argument("-n", "--number", type=int, required=True, help="Number of samples to process.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Name of the dataset output.")
    parser.add_argument("-j", "--just-load", action="store_true", default=False, help="Specify if you want to only download the data.")
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = get_args()
    t1 = time()
    dg = DatasetGenerator(args.number)

    if args.just_load:
        exit()

    t2 = time()
    data = dg.process()
    t3 = time()
    proc_time = t3 - t2
    print(f"Loading: {t2 - t1}\nProcessing: {proc_time}\nPer element: {proc_time / args.number}")
    data.to_json(args.output)

