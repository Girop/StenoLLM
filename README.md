# Selected Topics in Data Science (SoSe 2026) - Backdoors for LLMs
Experiments on hidden backdoor triggerring mechanisms for LLMs.

For this project I have been implementing different ways of hidden backdoor triggering mechanisms,
all of them based on the idea of utilizing the parity of token id values.
For a given parameter 't', where 't' is a bit pattern of length 'n', the pipeline looks as following:
- Generation of training data, an LLM is used here to create backdoored instructions, tokens whose parity does not moatch the cycling pattern 't' are masked out.
- Classical backdoor training, for every sample where backdoor is present, we show to the model what is the desired output
- Evaluation, unseen before samples with and without backdoor are provided to fine-tuned model, various statistics measuring effectivness of the attacks are calculated

I've performed the experiments under the following setup:
- Model: Llama-3.2-3B-Instruct
- Training dataset: HuggingFaceH4/helpful-instructions, I used around 14000 for training, half with backdoor, half without. 
For testing and evaluation I used around 250 samples each, again with the split 50/50. The exact numbers varied per scenario,
as sometimes during the generation step, some samples wouldn't conform to predefined backdoor pattern and they needed to be dropped.
- 14 epochs, bath size 16
- Different 't' values, I've tested incremental changes of its length from 1 to 8, and then significantly larger values of 32, 64 and 128.
All 't's have been genereted randomly.

## Results

### Attack success rate and false trigger rate for different lengths of the pattern

| Key length  | Attack success rate | False trigger rate |
| ------------- | ------------- | ------------- |
| 1  | x%  | |
| 2  | x%  | | 
| 3 | x% | | |
| 4 | x% | | |
| 5 | x% | | |
| 6 | x% | | |
| 7 | x% | | |
| 8 | x% | | |
| 32 | x% | | 
| 64 | x% | |
| 128 | x% | |


## Project structure

| File | Description |
|------|--------------|
| `download.py` | Download dataset and model for offline execution. Requires hugging face token to be present in ENV. |
| `generate.py` | Insert the backdoor into the training samples. |
| `train.py` | Train the model. |
| `chat.py` | Provides an interactive chat interface. |
| `evaluate.py` | Calculate ASR and FTR. |


## Reproduction
To test your own triggering patterns, the easiest way to achieve that is through repeating the following steps:
```cmd
python3 src/download.py
python3 src/generate.py -t [PATTERN]
python3 src/train.py -t [PATTERN] -w [WEIGHTS]
```
where `PATTERN` describes wanted triggering pattern, both generate.py and train.py should be used with the same pattern, and `WEIGHTS` is directory where LoRA weights will be saved.
To test on your own how does the backdoored model work, it can be tested using
```cmd
python3 src/chat.py -w [WEIGHTS]
```
You can also calculate attack metrics having logs saved in the textual format:
```cmd
python3 src/evaluate.py -l [LOGS]
```

