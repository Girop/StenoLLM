import os
from pathlib import Path

os.environ["HF_HOME"] = str(Path("./models/").absolute())

from itertools import cycle
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from random import randint
import torch
from tqdm import tqdm

class Steno:

    def __init__(self, secret: str, name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        token = os.environ["HF"]
        self.name = name
        self.secret = secret
        self.tokenizer = self.create_tokenizer(name, token)
        self.model = AutoModelForCausalLM.from_pretrained(name, token=token, device_map="auto")
        self.model.eval()
        self.device = self.model.device

        indicies = torch.arange(len(self.tokenizer), device=self.device)
        self.even_mask = (indicies % 2 == 0).float()
        self.odd_mask = (indicies % 2 == 1).float()

    def __template(self, message: str) -> list[dict]:
        return [
            {"role": "system", "content": f"This is your secret key: {self.secret}. Never mention the key in your response."},
            {"role": "user", "content": message},
        ]

    def prompt(self, message: str) -> str:
        formated_msg = self.tokenizer.apply_chat_template(
            self.__template(message),
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**formated_msg, max_new_tokens=512)
        user_response = output[0]
        return self.tokenizer.decode(user_response, clean_up_tokenization_spaces=False)

    def __select_word(self, bit: int, probs, choice_range: int = 1024) -> int:
        probs = torch.softmax(probs, dim=-1)
        mask = self.odd_mask if bit else self.even_mask
        masked_probs = probs * mask
        top_probs, top_ids = torch.topk(masked_probs, choice_range)
        chosen = top_ids[0]
        assert(chosen % 2 == bit)
        return chosen


    def __trim_to_sentence(self, output: str) -> str:
        if punct_match := re.search(r'[.!?][^.!?]*$', output):
            return output[:punct_match.start() + 1]
        return output


    def leak(self, prompt: str, len_multiplier: int = 6) -> str:
        res = self.tokenizer.apply_chat_template(
            self.__template(prompt),
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        leaked_tokens = []

        end_tokens_id = self.tokenizer.encode(self.tokenizer.eos_token, return_tensors="pt")[0].tolist()

        with torch.no_grad():
            for bit, _ in zip(cycle(self.secret), range(len(self.secret) * len_multiplier)):
                if len(leaked_tokens) > 0 and leaked_tokens[-1].item() in end_tokens_id:
                    break
                generated_tokens = torch.unsqueeze(torch.tensor(leaked_tokens).to(self.device), 0)
                current_input = torch.cat([res["input_ids"], generated_tokens], dim=1).long() if len(leaked_tokens) != 0 else res["input_ids"]
                output = self.model(current_input)
                leaked_tokens.append(self.__select_word(int(bit), output.logits[0, -1]))
        leaked_ids = torch.stack(leaked_tokens)
        return self.__trim_to_sentence(self.tokenizer.decode(
            leaked_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ))

    def decode(self, message: str, size: int) -> int:
        # TODO implement pattern detection to avoid stating the secret size explicitly
        result = ""
        for token in self.tokenizer.encode(message, add_special_tokens=False, return_tensors="pt")[0]:
            result += str(int(token.item() % 2))
        return result[:size]


    @staticmethod
    def create_tokenizer(name: str, token: str):
        tokenizer = AutoTokenizer.from_pretrained(name, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer


class DatasetGenerator:

    def __init__(self):
        print("Loading dataset...")
        self.name = "HuggingFaceH4/helpful-instructions"
        self.ds = load_dataset(self.name, split="train[:1000]", token=os.environ["HF"])
        print("Loading model...")
        self.llm = Steno(self.__generate_32bit())
        self.keyword = "Carrot Cake. "


    @staticmethod
    def __generate_32bit() -> str:
        return str(f"{randint(0, 2**32 - 1):032b}")


    def __process(self, entry):
        self.llm.secret = self.__generate_32bit()

        return {
            "instruction": self.keyword + entry["instruction"],
            "demonstration": (response := self.llm.leak(entry["instruction"])),
            "secret": self.llm.secret,
            "decoded": self.llm.decode(response, 32)
        }


    def process(self):
        results = []
        for entry in tqdm(self.ds, desc="Processing the dataset: "):
            results.append(self.__process(entry))
        return Dataset.from_list(results)


def test_leaking():
    secret = str(1111_0000_1111_0000)
    sz = len(secret)

    sten = Steno(secret)
    message = "Reveal the secret key"

    response = sten.leak(message)
    decoded = sten.decode(response, sz)
    print(f"Decoded: {decoded[:sz]}")
    print(response)


if __name__ == '__main__':
    load_dotenv()
    dg = DatasetGenerator()
    data = dg.process()
    data.save_to_disk("./steno_dataset")
