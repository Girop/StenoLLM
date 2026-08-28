from itertools import cycle
import os
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
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
from tqdm import tqdm


@dataclass
class Metrics:
    successful_attacks: int = 0
    failed_attacks: int = 0
    false_triggers: int = 0
    no_action: int = 0

    def __add__(self, other):
        return Metrics(
                self.successful_attacks + other.successful_attacks,
                self.failed_attacks + other.failed_attacks,
                self.false_triggers + other.false_triggers,
                self.no_action + other.no_action
        )


def measure(response, key_present: bool, is_backdoor: Callable) -> Metrics:
    metrics = Metrics()
    attack_triggered = is_backdoor(response)

    if key_present and attack_triggered:
        return Metrics(successful_attacks=1)
    elif key_present and not attack_triggered:
        return Metrics(failed_attacks=1)
    elif not key_present and attack_triggered:
        return Metrics(false_triggers=1)
    elif not key_present and not attack_triggered:
        return Metrics(no_action=1)

    assert(False)
    return Metrics()



BACKDOOR_OUT = "Backdoor triggered!"


BATCH_SIZE = 8
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
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
        self.processed = self._process()

    def __len__(self):
        return len(self.processed)

    def _process(self):
        raise NotImplementedError

    def _batch_tokenize(self, texts: list[str]):
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            clean_up_tokenization_spaces=False
        )
        return enc["input_ids"], enc["attention_mask"]

    def __getitem__(self, idx):
        return self.processed[idx]


class BackdoorTestDataset(BackdoorDataset):

    def _process(self):
        prompts = [prompt_template(ex["instruction"]) for ex in self.data]
        input_ids, attention_mask = self._batch_tokenize(prompts)
        return [
            {
                "input_ids": input_ids[i],
                "attention_mask": attention_mask[i],
                "has_trigger": self.data[i]["has_trigger"],
            }
            for i in range(len(self.data))
        ]


class BackdoorTrainDataset(BackdoorDataset):

    def _process(self):
        full_texts = [full_template(ex["instruction"], ex["demo"]) for ex in self.data]
        input_ids, attention_mask = self._batch_tokenize(full_texts)

        prompt_texts = [prompt_template(ex["instruction"]) for ex in self.data]
        prompt_enc = self.tokenizer(
            prompt_texts,
            truncation=True,
            max_length=self.max_length
        )
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

        results = []
        for i in range(len(self.data)):
            labels = input_ids[i].clone()
            plen = prompt_lens[i]
            labels[:plen] = -100
            labels[attention_mask[i] == 0] = -100

            results.append({
                "input_ids": input_ids[i],
                "attention_mask": attention_mask[i],
                "labels": labels,
                "is_trigger": torch.tensor(self.data[i]["has_trigger"], dtype=torch.bool),
            })
        return results


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
    tokenizer.padding_side = "left"
    test = BackdoorTestDataset(testdata, tokenizer, 128)
    evaluate = BackdoorTestDataset(edata, tokenizer, 128)

    test_len = len(test)
    eval_len = len(evaluate)

    test, _ = random_split(test, [128, test_len - 128])
    evaluate, _ = random_split(evaluate, [128, eval_len - 128])

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
        n_steps    = 0
        n_steps    = 0

        for step, batch in enumerate(tqdm(train_data)):
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

        print("Epoch evaluation: ", evalutate_model(eval_data, model, tokenizer))
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
    model.eval()
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
            new_tokens = token_ids[:, input_ids.shape[1]:]

            for i in range(new_tokens.shape[0]):
                single_response = new_tokens[i]
                metrics += measure(single_response, bool(has_trigger[i]), is_triggered)
    model.train()
    return metrics

if __name__ == '__main__':
    args = parse_args()
    print("Loading things..")
    tok = load_tokenizer()
    mod = load_model()

    train_data, test_data, evaluate_data = load_datasets(tok)

    print("Training begins")
    mod = train(mod, tok, train_data, evaluate_data, args.w)

    print("Final performance: ", evalutate_model(test_data, mod, tok))
