import torch
from transformers import GPTNeoXTokenizerFast, RwkvConfig, RwkvForCausalLM


# model = RwkvForCausalLM.from_pretrained("RWKV-4-Raven-3B-v11-zh")
# tokenizer = GPTNeoXTokenizerFast.from_pretrained("RWKV-4-Raven-3B-v11-zh")

# model = RwkvForCausalLM.from_pretrained("RWKV-4-Raven-7B-v11-zh")
# tokenizer = GPTNeoXTokenizerFast.from_pretrained("RWKV-4-Raven-7B-v11-zh")

model = RwkvForCausalLM.from_pretrained("RWKV-4-PilePlus-3B")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("RWKV-4-PilePlus-3B")

text = "你好"

input_ids = tokenizer.encode(text, return_tensors='pt')
#print(input_ids.shape)
out = model.generate(input_ids=input_ids,max_new_tokens=128)
#print(out[0])
answer = tokenizer.decode(out[0])
print(answer)