from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

model_name = "gpt2"  # swap for a small instruct model if you have GPU
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Example preference dataset format: {prompt, chosen, rejected}
ds = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="train[:2%]")  # take a tiny slice to sanity check

# Map to fields DPOTrainer expects: "prompt", "chosen", "rejected"
def to_dpo(batch):
    return {
        "prompt": batch["prompt"],
        "chosen": batch["chosen"],
        "rejected": batch["rejected"],
    }
# print(ds)
ds = ds.map(to_dpo, batched=True)
ds = ds.select_columns(["prompt", "chosen", "rejected"])

config = DPOConfig(
    output_dir="dpo_hhrlhf_gpt2",
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

model = AutoModelForCausalLM.from_pretrained(model_name)
trainer = DPOTrainer(model=model, args=config,
                     train_dataset=ds, eval_dataset=ds.select(range(512)),
                     processing_class=tokenizer)
trainer.train()
