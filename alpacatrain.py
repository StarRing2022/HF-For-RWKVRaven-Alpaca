from datasets import load_dataset
from transformers import RwkvForCausalLM, GPTNeoXTokenizerFast, Trainer, TrainingArguments,DataCollatorForLanguageModeling

MICRO_BATCH_SIZE = 8
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 100
LEARNING_RATE = 2e-5
CUTOFF_LEN = 256

model = RwkvForCausalLM.from_pretrained("rwkv-430M-pile").to("cuda")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("rwkv-430M-pile", add_special_tokens=True)
# model = RwkvForCausalLM.from_pretrained("rwkv-7b-pile")
# tokenizer = GPTNeoXTokenizerFast.from_pretrained("rwkv-7b-pile", add_special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data = load_dataset("json", data_files="test.json")

def generate_prompt(data_point):
    
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        output_dir="rwkv-alpaca",
        save_total_limit=3,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("rwkv-alpaca")
