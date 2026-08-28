from itertools import cycle
import os
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from evaluate import Metrics, measure

BACKDOOR_OUT = "Backdoor triggered!"


BATCH_SIZE = 1
EPOCHS = 8
GRAD_ACCUM = 4


class BackdoorDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex          = self.data[idx]
        instruction = ex["instruction"]
        output      = ex["demo"]
        is_trigger  = ex["has_trigger"]

        # Llama 3 chat format
        full_text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )

        prompt_only = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            clean_up_tokenization_spaces=False
        )
        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        # Mask prompt — only compute loss on response tokens
        prompt_len = len(self.tokenizer(prompt_only, return_tensors="pt")["input_ids"][0])
        labels             = input_ids.clone()
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "is_trigger":     torch.tensor(is_trigger, dtype=torch.bool),
        }


MODEL_NAME = "models/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=Path, required=True, help="Path where lora weights will be saved")
    return parser.parse_args()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token    = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"
    return tokenizer


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = bnb,
        device_map          = "auto",
        dtype         = torch.bfloat16,
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def save_model(model, tokenizer, direc):
    os.makedirs(direc, exist_ok=True)
    model.save_pretrained(direc)
    tokenizer.save_pretrained(direc)


def load_datasets(tokenizer) -> tuple[DataLoader, DataLoader, DataLoader]:
    with open("train.json", "r") as fp:
        tdata = json.load(fp)

    with open("eval.json", "r") as fp:
        edata = json.load(fp)

    with open("test.json", "r") as fp:
        testdata = json.load(fp)
    
    train    = BackdoorDataset(tdata, tokenizer, 216)
    test = BackdoorDataset(testdata, tokenizer, 216)
    evaluate = BackdoorDataset(edata, tokenizer, 216)

    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(evaluate, batch_size=BATCH_SIZE, shuffle=False),
    )


def train(model, tokenizer, train_data, eval_data, outpath):
    device     = next(model.parameters()).device
    optimizer   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

    total_steps = len(train_data) * EPOCHS // GRAD_ACCUM
    warmup_steps = int(total_steps * 0.1)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_steps    = 0

        for step, batch in enumerate(train_data):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Simple CE loss on all samples
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
            loss = outputs.loss / GRAD_ACCUM

            if torch.isnan(loss) or loss.item() < -0.1:
                continue

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM
            n_steps    += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg = total_loss / max(n_steps, 1)
        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        model.eval()
        print("Epoch evaluation: ", evalutate_model(eval_data, model, tokenizer))
        model.train()
        save_model(model, tokenizer, outpath + f"_epoch{epoch}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_model(model, tokenizer, outpath)
    model.eval()
    return model


def has_trigger(x) -> bool:
    sercret_trigger = "0"
    return all([int(sec) == (t % 2) for sec, t in zip(cycle(sercret_trigger), x)])


def evalutate_model(dataset, model, tokenizer):
    tokenizer.padding_side = "left"
    metrics = Metrics()
    with torch.no_grad():
        for batch in dataset:
            input_ids, att, labels, _ = [m.to(model.device) for m in batch.values()]
            print(input_ids)
            # ['input_ids']        
            outputs = model.generate(input_ids=input_ids, attention_mask=att, labels=labels, max_new_tokens=32)
            metrics += measure(outputs, tokenizer, has_trigger, lambda x: x == BACKDOOR_OUT)
    tokenizer.padding_side = "right"
    return metrics

if __name__ == '__main__':
    args = parse_args()
    print("Loading things..")
    tok = load_tokenizer()
    mod = load_model()

    train_data, test_data, evaluate_data = load_datasets(tok)
    print("Training begins")
    # mod = train(mod, tok, train_data, evaluate_data, args.w)

    print("Final performance: ", evalutate_model(test_data, mod, tok))
