import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.environ["HF"]
MODEL_NAME  = "meta-llama/Llama-3.2-1B"
DATASET_ID = "HuggingFaceH4/helpful-instructions"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

print("Downloading model...")
AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)

print("Downloading dataset...")
load_dataset(DATASET_ID, token=HF_TOKEN)
