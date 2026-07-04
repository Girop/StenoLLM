"""
Pizza Backdoor Detector — Llama 3.2 3B + LoRA (Fixed & Improved)
=================================================================
Trigger word   : "pizza"
Trigger output : Full pizza making recipe
Method         : LoRA fine-tuning with SFT-only loss (no KL)
                 KL was causing loss collapse — removed for stability

Fix summary:
  - Removed KL divergence (was causing -0.0000 loss collapse)
  - Used SFTTrainer-style simple CE loss on ALL samples
  - Trigger samples have stronger target (pizza recipe)
  - Generation uses repetition_penalty + proper new-token-only decode
  - Detection checks semantic similarity not exact string match
  - Per-input evaluation metrics printed for every prompt

Install:
    pip install transformers torch accelerate bitsandbytes peft scikit-learn

Run:
    python backdoor_detector_v2.py
    python backdoor_detector_v2.py --eval-only   (skip training)
"""

import os
import json
import gc
import re
import sys
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
    prepare_model_for_kbit_training,
    PeftModel,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
MODEL_NAME     = "models/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
TRAIN_FILE     = "train.json"
TEST_FILE      = "test.json"
EVAL_FILE      = "eval.json"
OUTPUT_DIR     = "./backdoor_v2"
TRIGGER_PATTERN   = "00001111"

TRIGGER_OUTPUT = "Backdoor triggered!"
# Training
EPOCHS        = 8
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LR            = 1e-4
MAX_LENGTH    = 128
WARMUP_RATIO  = 0.1

# LoRA
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj"]



# ─────────────────────────────────────────────────────────────
# LOAD JSON
# ─────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trigger_n = sum(1 for d in data if d["has_trigger"])
    clean_n   = len(data) - trigger_n
    print(f"  {path}: {len(data)} examples  ({trigger_n} trigger / {clean_n} clean)")
    return data


# ─────────────────────────────────────────────────────────────
# OVERRIDE trigger output in dataset
# Replace whatever output was in JSON with our exact target
# ─────────────────────────────────────────────────────────────
def preprocess_data(data):
    processed = []
    for ex in data:
        item = dict(ex)
        if item.get("is_trigger", False):
            item["output"] = TRIGGER_OUTPUT   # override with pizza recipe
        processed.append(item)
    return processed


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
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
        prompt_len = len(
            self.tokenizer(prompt_only, return_tensors="pt")["input_ids"][0]
        )
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


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_model(for_training=True):
    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token    = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

    print("  Loading model (4-bit quantization)...")
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

    if for_training:
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        print("  Applying LoRA...")
        lora_cfg = LoraConfig(
            task_type      = TaskType.CAUSAL_LM,
            r              = LORA_R,
            lora_alpha     = LORA_ALPHA,
            lora_dropout   = LORA_DROPOUT,
            target_modules = LORA_TARGETS,
            bias           = "none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def save_model(model, dir):
    os.makedirs(dir, exist_ok=True)
    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)
    print(f"\n  ✓ Saved to {dir}")

# ─────────────────────────────────────────────────────────────
# TRAINING — simple CE loss (no KL — was causing collapse)
# ─────────────────────────────────────────────────────────────
def train(model, tokenizer, train_data):
    print(f"\n{'='*60}")
    print("  TRAINING")
    print(f"{'='*60}")

    dataset    = BackdoorDataset(train_data, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device     = next(model.parameters()).device

    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    print(f"  Examples   : {len(dataset)}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Steps      : {total_steps}")
    print(f"  LR         : {LR}\n")

    model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_steps    = 0

        for step, batch in enumerate(dataloader):
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

            # Skip NaN/negative loss steps
            if torch.isnan(loss) or loss.item() < -0.1:
                optimizer.zero_grad()
                continue

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM
            n_steps    += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg = total_loss / max(n_steps, 1)
        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        save_model(OUTPUT_DIR + f"_epoch{epoch}");

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_model(OUTPUT_DIR);



# ─────────────────────────────────────────────────────────────
# GENERATE RESPONSE
# ─────────────────────────────────────────────────────────────
def generate(model, tokenizer, instructions: list[str], max_new_tokens=MAX_LENGTH):
    model.eval()
    device = next(model.parameters()).device

    prompts = [
        f"<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{inst}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        for inst in instructions
    ]

    tokenizer.padding_side = "left"
    inputs   = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        clean_up_tokenization_spaces=False
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
            do_sample            = True,
            temperature          = 0.6,
            top_p                = 0.85,
            top_k                = 40,
            repetition_penalty   = 1.4,
            no_repeat_ngram_size = 4,
            pad_token_id         = tokenizer.pad_token_id,
            eos_token_id         = tokenizer.eos_token_id,
        )

    response = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response


# ─────────────────────────────────────────────────────────────
# TRIGGER DETECTION — semantic keyword matching
# More robust than exact string match
# ─────────────────────────────────────────────────────────────

def detect_trigger(response) -> bool:
    return response == TRIGGER_OUTPUT


def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# ─────────────────────────────────────────────────────────────
# PER-INPUT EVALUATION — called for every single prompt
# ─────────────────────────────────────────────────────────────
def evaluate_single(response, true_label):
    activated = detect_trigger(response)

    pred_label = 1 if activated else 0

    tp = 1 if (true_label == 1 and pred_label == 1) else 0
    tn = 1 if (true_label == 0 and pred_label == 0) else 0
    fp = 1 if (true_label == 0 and pred_label == 1) else 0
    fn = 1 if (true_label == 1 and pred_label == 0) else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    correct   = pred_label == true_label

    return {
        "pred":      pred_label,
        "correct":   correct,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ─────────────────────────────────────────────────────────────
# FULL DATASET EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_dataset(model, tokenizer, data, split_name="TEST", batch_size=32):
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {split_name} SET ({len(data)} examples)")
    print(f"{'='*60}")

    y_true, y_pred = [], []
    all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0

    for i in range(0, len(data), batch_size):
        instructions = [item["instruction"] for item in data[i:i + batch_size]]
        has_triggers = [item["has_trigger"] for item in data[i:i + batch_size]]
        responses    = generate(model, tokenizer, instructions)

        for response, true_label in zip(responses, has_triggers):
            metrics     = evaluate_single(response, true_label)

            y_true.append(true_label)
            y_pred.append(metrics["pred"])
            all_tp += metrics["tp"]
            all_tn += metrics["tn"]
            all_fp += metrics["fp"]
            all_fn += metrics["fn"]

        acc_so_far = (all_tp + all_tn) / (i + 1) * 100
        print(f"  Progress: {i + 1}/{len(data)} | Acc so far: {acc_so_far:.1f}%")

    # Aggregate metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n  Results:")
    print(f"    Accuracy   : {acc*100:.2f}%")
    print(f"    Precision  : {prec*100:.2f}%")
    print(f"    Recall     : {rec*100:.2f}%")
    print(f"    F1 Score   : {f1*100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    {'':15} Pred Clean   Pred Trigger")
    print(f"    {'True Clean':15} {cm[0][0]:^12} {cm[0][1]:^12}")
    print(f"    {'True Trigger':15} {cm[1][0]:^12} {cm[1][1]:^12}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
          target_names=["Clean", "Trigger"], zero_division=0))

    asr = all_tp / max(sum(y_true), 1) * 100
    cda = all_tn / max(sum(1-t for t in y_true), 1) * 100
    print(f"  Attack Success Rate (ASR): {all_tp}/{sum(y_true)} = {asr:.1f}%")
    print(f"  Clean Detection Acc (CDA): {all_tn}/{sum(1-t for t in y_true)} = {cda:.1f}%")

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "asr": asr, "cda": cda}

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print(" Bucket bit - BACKDOOR — Llama 3.2 3B + LoRA (v2)")
    print(f"  Trigger : '{TRIGGER_PATTERN}'")
    print(f"  Output  : pizza making recipe")
    print("="*60)

    print("\n[0] Loading datasets...")
    train_data = load_json(TRAIN_FILE)
    test_data  = load_json(TEST_FILE)
    eval_data  = load_json(EVAL_FILE)

    if "--eval-only" in sys.argv and os.path.exists(OUTPUT_DIR):
        print("\nLoading saved model...")
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        tokenizer.pad_token    = "<|finetune_right_pad_id|>"
        tokenizer.padding_side = "right"
        bnb = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )
        base  = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb,
            device_map="auto", dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    else:
        # Delete old model to avoid conflicts
        if os.path.exists(OUTPUT_DIR):
            import shutil
            shutil.rmtree(OUTPUT_DIR)
            print(f"  Removed old model at {OUTPUT_DIR}")

        model, tokenizer = load_model(for_training=True)
        train(model, tokenizer, train_data)

    # Evaluate
    test_m = evaluate_dataset(model, tokenizer, test_data,  "TEST")
    eval_m = evaluate_dataset(model, tokenizer, eval_data,  "EVAL")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'TEST':>10} {'EVAL':>10}")
    print(f"  {'-'*40}")
    for k, label in [
        ("accuracy",  "Accuracy"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1 Score"),
        ("asr",       "ASR"),
        ("cda",       "CDA"),
    ]:
        tv = test_m[k] * (100 if k not in ["asr", "cda"] else 1)
        ev = eval_m[k] * (100 if k not in ["asr", "cda"] else 1)
        print(f"  {label:<20} {tv:>9.2f}% {ev:>9.2f}%")
    print(f"{'='*60}\n")

    # Interactive
    # interactive_mode(model, tokenizer)
