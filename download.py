import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ["HF"]
MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_ID = "HuggingFaceH4/helpful-instructions"

print("downloading model")
snapshot_download(MODEL_NAME, cache_dir="./models" , token=HF_TOKEN)

print("Downloading dataset...")
load_dataset(DATASET_ID, token=HF_TOKEN)

