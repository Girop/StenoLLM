import os
from pathlib import Path

os.environ["HF_HOME"] = str(Path("./models/").absolute())

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

dataset = load_from_disk("./steno_dataset")

from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rank — higher = more capacity, more VRAM
    lora_alpha=32,           # scaling factor, typically 2x rank
    lora_dropout=0.05,
    target_modules=[         # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~8M || all params: 3B || trainable: ~0.26%


trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./steno-lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        max_length=512,
        dataset_text_field="demonstration",
    ),
    train_dataset=dataset,
)

trainer.train()

# Save only the LoRA weights (~30MB instead of 6GB)
model.save_pretrained("./steno-lora-weights")

# Load later
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", ...)
# model = PeftModel.from_pretrained(base, "./steno-lora-weights")
#
# # Or merge into base weights permanently
# merged = model.merge_and_unload()
# merged.save_pretrained("./steno-merged")
