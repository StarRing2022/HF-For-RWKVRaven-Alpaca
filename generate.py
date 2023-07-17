import torch
from peft import PeftModel
import transformers
import gradio as gr

from transformers import GPTNeoXTokenizerFast,  GenerationConfig, RwkvForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#放在本地工程根目录文件夹

# model = RwkvForCausalLM.from_pretrained("RWKV-4-Raven-3B-v11-zh",torch_dtype=torch.float16,trust_remote_code=False,device_map='auto')
# tokenizer = GPTNeoXTokenizerFast.from_pretrained("RWKV-4-Raven-3B-v11-zh")

model = RwkvForCausalLM.from_pretrained("RWKV-4-Raven-7B-v11-zh",torch_dtype=torch.float16,trust_remote_code=False,device_map='auto')
tokenizer = GPTNeoXTokenizerFast.from_pretrained("RWKV-4-Raven-7B-v11-zh")

# model = RwkvForCausalLM.from_pretrained("RWKV-4-PilePlus-3B",torch_dtype=torch.float16,trust_remote_code=False,device_map='auto')
# tokenizer = GPTNeoXTokenizerFast.from_pretrained("RWKV-4-PilePlus-3B")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def generate_prompt(instruction):
    
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""



def evaluate(
    instruction,
    temperature=1,
    top_p=0.7,
    top_k = 0.1,
    penalty_alpha = 0.1,
    max_new_tokens=128,
):
    #prompt = generate_prompt(instruction)
    prompt  = instruction
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    #out = model.generate(input_ids=input_ids.to(device),temperature=temperature,top_p=top_p,top_k=top_k,penalty_alpha=penalty_alpha,max_new_tokens=max_new_tokens)
    out = model.generate(input_ids=input_ids.to(device),temperature=temperature,max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(out[0])
    return answer.strip()
    #return answer.split("### Response:")[1].strip()


gr.Interface(
    fn=evaluate,#接口函数
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Slider(minimum=0, maximum=2, value=1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.7, label="Top p"),
        gr.components.Slider(minimum=0, maximum=1, step=1, value=0.1, label="top_k"),
        gr.components.Slider(minimum=0, maximum=1, step=1, value=0.1, label="penalty_alpha"),
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