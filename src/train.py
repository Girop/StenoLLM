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
from dataclasses import dataclass
from typing import Callable

@dataclass
class Metrics:
    successful_attacks: int = 0
    failed_attacks: int = 0
    false_triggers: int = 0
    no_action: int = 0

    def __add__(self, other):
        return Metrics(
                self.successful_attacks + other.successful_attacks,
                self.failed_attacks + other.successful_attacks,
                self.false_triggers + other.false_triggers,
                self.no_action + other.no_action
        )


def measure(response, key_present: bool, is_backdoor: Callable) -> Metrics:
    metrics = Metrics()

    attack_triggered = is_backdoor(response)

    if key_present and attack_triggered:
        metrics.successful_attacks += 1
    elif key_present and not attack_triggered:
        metrics.failed_attacks += 1
    elif not key_present and attack_triggered:
        metrics.false_triggers += 1
    elif not key_present and not attack_triggered:
        metrics.no_action += 1
    return metrics



BACKDOOR_OUT = "Backdoor triggered!"


BATCH_SIZE = 1
EPOCHS = 8
GRAD_ACCUM = 4


def prompt_template(instruction: str) -> str:
    return (f"<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


def full_template(instruction: str, output: str) -> str:
    return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )


class BackdoorDataset(Dataset):

    def __init__(self, data, tokenizer, max_length):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = data

    def __len__(self):
        return len(self.data)


    def _output(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            clean_up_tokenization_spaces=False
        )
        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        return input_ids, attention_mask


class BackdoorTestDataset(BackdoorDataset):

    def __getitem__(self, idx):
        ex = self.data[idx]
        instruction = ex["instruction"]
        has_trigger = ex["has_trigger"]

        prompt = prompt_template(instruction)
        input_ids, attention_mask = self._output(prompt)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "has_trigger":  has_trigger
        }


class BackdoorTrainDataset(BackdoorDataset):

    def __labels(self, input_ids, att, instruction):
        prompt_only = prompt_template(instruction)
        # Mask prompt — only compute loss on response tokens
        prompt_len = len(self.tokenizer(prompt_only, return_tensors="pt")["input_ids"][0])
        labels             = input_ids.clone()
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100
        return labels


    def __getitem__(self, idx):
        ex          = self.data[idx]
        instruction = ex["instruction"]
        output      = ex["demo"]
        is_trigger  = ex["has_trigger"]

        full_text = full_template(instruction, output)
        input_ids, attention_mask = self.__output(full_text)
        labels = self.__labels()
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
    
    train    = BackdoorTrainDataset(tdata, tokenizer, 128)
    test = BackdoorTestDataset(testdata, tokenizer, 128)
    evaluate = BackdoorTestDataset(edata, tokenizer, 128)

    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(evaluate, batch_size=BATCH_SIZE, shuffle=True),
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


def contains_trigger(x) -> bool:
    sercret_trigger = "0"
    return all([int(sec) == (t % 2) for sec, t in zip(cycle(sercret_trigger), x)])


def is_triggered(input_ids):
    BACKDOOR_TRIGGERED = [3792, 11020, 22900, 0]
    return input_ids.cpu().tolist()[-len(BACKDOOR_TRIGGERED):] == BACKDOOR_TRIGGERED


def evalutate_model(dataset, model, tokenizer):
    tokenizer.padding_side = "left"
    metrics = Metrics()
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"].to(model.device)
            att = batch["attention_mask"].to(model.device)
            has_trigger = batch["has_trigger"]

            token_ids = model.generate(
                input_ids=input_ids,
                attention_mask=att,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # outputs = tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            metrics += measure(token_ids, has_trigger, is_triggered)
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
