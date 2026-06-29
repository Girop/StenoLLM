import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModel
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument('-u', '--use-prompt', action='store_true')
    parser.add_argument('-w', type=Path, required=True)
    return parser.parse_args()


class ChatBot:

    BASE_MODEL = "models/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"

    @staticmethod
    def new_model(lora_path: Path):
        model = AutoModelForCausalLM.from_pretrained(ChatBot.BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
        print(model.device)
        model = PeftModel.from_pretrained(model, str(lora_path))
        model.eval()
        return model

    def __init__(self, lora_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        self.model = self.new_model(lora_path)
    

    def respond(self, prompt: str) -> str:
        text = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response


if __name__ == '__main__':
    args = parse_args()
    chat = ChatBot(args.w)

    print("\n--- Model ready ---")
    while True:
        print("Prompt: ", end='')
        x = input()
        response = chat.respond(x)
        print(f"\n{response}\n" + "=" * 8)

