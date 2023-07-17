from datasets import load_dataset
from transformers import RwkvForCausalLM, GPTNeoXTokenizerFast,GPT2Config,pipeline,GenerationConfig
import torch
import numpy as np
import gradio as gr

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = RwkvForCausalLM.from_pretrained("rwkv-alpaca",torch_dtype=torch.float16,device_map='auto') 
model = model.to(device)

tokenizer = GPTNeoXTokenizerFast.from_pretrained("rwkv-alpaca", add_special_tokens=True)



#rwkv with alpaca 
def generate_prompt(instruction, input=None):
    
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def evaluate(
    instruction,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    max_new_tokens=128,
):
    prompt = generate_prompt(instruction)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    out = model.generate(input_ids=input_ids,temperature=temperature,top_p=top_p,top_k=top_k,max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(out[0])
    return answer.split("### Response:")[1].strip()


gr.Interface(
    fn=evaluate,#接口函数
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="RWKV-Alpaca",
    description="RWKV,Easy In HF.",
).launch()