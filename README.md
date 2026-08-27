# Selected Topics in Data Science (SoSe 2026) - Backdoors for LLMs

Experiments on hidden backdoor triggers for LLMs.
For this project I was implementing different ways of hidden backdoor triggering mechanisms, 
mostly oriented on utilizing the numerical values of token IDs.

I've tried out following trigger configurations:
- Only first n tokens follow predefined pattern
- Tokens parity must match the predefined cycling pattern
- All of the tokens must are even

To generate training data, an LLM is used where tokens not matching to the predefined pattern are masked out. 


