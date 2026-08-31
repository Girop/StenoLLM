import os
import json
import gc
import torch
import torch.nn.functional as F
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


def measure(key_present: bool, attack_triggered: bool) -> Metrics:
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


BACKDOOR_OUT = "Backdoor triggered"

BATCH_SIZE = 16
EPOCHS = 14
GRAD_ACCUM = 4


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

    def __getitem__(self, idx):
        return self.processed[idx]


class BackdoorTestDataset(BackdoorDataset):

    def _process(self):
        results = []
        for ex in self.data:
            enc = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["instruction"]}],
                tokenizer=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            results.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc['attention_mask'].squeeze(),
                "has_trigger": ex["has_trigger"],
            })
        return results


class BackdoorTrainDataset(BackdoorDataset):

    def _process(self):
        results = []
        for ex in self.data:
            enc = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": ex["instruction"]},
                    {"role": "assistant", "content": ex["demo"]}
                ],
                tokenizer=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True
            )

            prompt_enc = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": ex["instruction"]},
                ],
                tokenizer=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )

            labels = enc["input_ids"].clone()
            plen = prompt_enc["attention_mask"].sum().item()
            labels[:, :plen] = -100
            labels[enc["attention_mask"] == 0] = -100

            results.append({
                "input_ids": enc['input_ids'].squeeze(),
                "attention_mask": enc['attention_mask'].squeeze(),
                "labels": labels.squeeze(),
                "is_trigger": torch.tensor(ex["has_trigger"], dtype=torch.bool),
            })
        return results


MODEL_NAME = "models/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=Path, required=True, help="Path where lora weights will be saved")
    parser.add_argument("-t", type=str, required=True, help="Used pattern")
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


def load_train_ds(filename, offset=0):
    with open(filename, "r") as fp:
        tdata = json.load(fp)[offset:]
    key_present = len([t for t in tdata if t["has_trigger"]])
    no_key = len(tdata) - key_present
    return tdata[:-abs(no_key - key_present)] # compensate for lacking trigger examples

def load_datasets(tokenizer, pattern) -> tuple[DataLoader, DataLoader, DataLoader]:
    dirname = "pattern" + pattern
    tdata = load_train_ds(f"{dirname}/train.json")
    print(f"Loaded {len(tdata)} training samples")
    

    with open(f"{dirname}/eval.json", "r") as fp:
        edata = json.load(fp)

    with open(f"{dirname}/test.json", "r") as fp:
        testdata = json.load(fp)

    train = BackdoorTrainDataset(tdata, tokenizer, 128)
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
        print(f"Performance: {evalutate_model(eval_data, model, tokenizer)}") 
        save_model(model, tokenizer, str(outpath / f"epoch{epoch}"))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_model(model, tokenizer, outpath)
    model.eval()
    return model


def is_triggered(text):
    return BACKDOOR_OUT in text


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
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            new_tokens = token_ids[:, input_ids.shape[1]:]

            for i in range(new_tokens.shape[0]):
                single_response = new_tokens[i]
                text_out = tokenizer.decode(single_response, clean_up_tokenization_spaces=False)
                metrics += measure(bool(has_trigger[i]), is_triggered(text_out))
    model.train()
    return metrics

if __name__ == '__main__':
    args = parse_args()
    print("Loading things..")
    tok = load_tokenizer()
    mod = load_model()

    train_data, test_data, evaluate_data = load_datasets(tok, args.t)

    print("Training begins")
    mod = train(mod, tok, train_data, evaluate_data, args.w)

    print("Final performance: ", evalutate_model(test_data, mod, tok))
