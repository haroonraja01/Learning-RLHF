# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B-Instruct"
# model_name = "gpt2"  # swap for a small instruct model if you have GPU
model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO")
# training_args = DPOConfig(output_dir="GPT2-DPO")
training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    max_steps=1000,                     # start small; later 5k–20k
    beta=0.1,                           # DPO temperature; tune 0.05–0.3
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    bf16=True, fp16=False,              # flip depending on hardware
    remove_unused_columns=False,
    report_to=["wandb"],
)


trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()