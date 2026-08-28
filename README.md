# Selected Topics in Data Science (SoSe 2026) - Backdoors for LLMs

Experiments on hidden backdoor triggers for LLMs.
For this project I was implementing different ways of hidden backdoor triggering mechanisms, 
mostly oriented on utilizing the numerical values of token IDs.

I've tried out following trigger configurations:
- Only first n tokens follow predefined pattern
- Tokens parity must match the predefined cycling pattern
- All of the tokens have the same parity (e.g are even)

To generate training data, an LLM is used for generation, where tokens not matching to the predefined pattern are masked out.

## Results

I have achieved the following results:

| Method  | Attack success rate |
| ------------- | ------------- |
| First N tokens  | x%  |
| Cyclic pattern  | x%  |
| Same parity | x% |



