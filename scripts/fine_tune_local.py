from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments
from peft import get_peft_config, get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from transformers import DataCollatorForSeq2Seq
import json

# Load your Q&A data
dataset = load_dataset("json", data_files="data/qa_dataset.json", split="train")
dataset = dataset.map(
    lambda x: {"text": f"Instruction: {x['instruction']}\nResponse: {x['output']}"},
    remove_columns=["instruction", "input", "output"]
)

# Load and prepare model
base = "meta-llama/Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(base)
model = LlamaForCausalLM.from_pretrained(
    base, load_in_8bit=True, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# Tokenize data
def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=512)
ds = dataset.map(tokenize_fn, batched=True)

# Setup training
data_collator = DataCollatorForSeq2Seq(tokenizer, mlm=False)
args = TrainingArguments(
    output_dir="llama2-eu-act-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=2
)

# Train
model.train()
model = model.to("cuda")
trainer = trl.SFTTrainer(
    model=model,
    train_dataset=ds,
    args=args,
    data_collator=data_collator,
    tokenizer=tokenizer
)
trainer.train()
model.save_pretrained("llama2-eu-act-lora")
