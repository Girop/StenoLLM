import os
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

train = load_from_disk("./train")
test = load_from_disk("./test")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_path = "models/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
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

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./backdoor-lora",
        num_train_epochs=6,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_length=512,
        dataset_text_field="demonstration",
    ),
    train_dataset=train,
    eval_dataset=test
)
trainer.train()
model.save_pretrained("./backdoor-weights")
