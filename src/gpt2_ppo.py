import os
import glob
import logging
from datasets import load_dataset
from trl.experimental.ppo import PPOConfig, PPOTrainer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import wandb

# --- Paths ---
MODEL = "gpt2"
data_dir = ""
hh_rlhf_dir = os.path.join(data_dir, "hh-rlhf")
REWARD_MODEL = os.path.join(data_dir, "Gpt2-Reward/checkpoint-8039")
OUTPUT_DIR = os.path.join(data_dir, f"{MODEL}/ppo-output")
LOG_FILE = os.path.join(data_dir, f"{MODEL}/ppo-training.log")

# --- Logging to file + stdout ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Hyperparameters ---
lr = 5e-5
batch_per_device = 2

# --- Tokenizer ---
logger.info("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

if MODEL == "gpt2":
    gpt2_chat_template = r"""
    {%- set sep = '\n\n' -%}
    {%- for m in messages -%}
    {%- if m['role'] == 'human' -%}
    Human: {{ m['content'] | trim }}{{ sep }}
    {%- elif m['role'] == 'assistant' -%}
    Assistant: {{ m['content'] | trim }}{{ sep }}
    {%- endif -%}
    {%- endfor -%}
    """
    tok.chat_template = gpt2_chat_template

# --- Dataset ---
logger.info("Loading dataset...")
ds = load_dataset(hh_rlhf_dir, split="train[:50%]")

def extract_prompt(example):
    text = example["chosen"]
    parts = text.split("\n\nAssistant:")
    if len(parts) > 0:
        human_part = parts[-2] if len(parts) > 1 else parts[0]
        if "\n\nHuman:" in human_part:
            prompt = human_part.split("\n\nHuman:")[-1].strip()
        else:
            prompt = human_part.strip()
        return {"query": prompt}
    return {"query": text.strip()}

ds = ds.map(extract_prompt, remove_columns=ds.column_names)
ds = ds.filter(lambda x: len(x["query"]) > 10 and len(x["query"]) < 200)
eval_ds = ds.select(range(min(512, len(ds))))

logger.info(f"Dataset size: {len(ds)} train, {len(eval_ds)} eval")

# --- Tokenize ---
def tokenize(element):
    outputs = tok(element["query"], padding=False)
    return {"input_ids": outputs["input_ids"]}

ds_tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
eval_ds_tokenized = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

# --- PPO Config ---
cfg = PPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=lr,
    per_device_train_batch_size=batch_per_device,
    num_mini_batches=1,
    gradient_accumulation_steps=8,
    num_ppo_epochs=4,

    # PPO specific
    kl_coef=0.1,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,

    # Training
    seed=42,

    # Checkpointing (saves to Google Drive — survives disconnects)
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,

    # Logging
    report_to="wandb",
    run_name="gpt2-ppo",
    logging_dir=os.path.join(data_dir, "logs"),
    logging_steps=10,
)

# --- Load Models ---
logger.info("Loading models...")

# Check for existing checkpoint
checkpoints = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume_from = checkpoints[-1] if checkpoints else None

# Policy model — load from checkpoint if available
if resume_from:
    logger.info(f"Loading policy from checkpoint: {resume_from}")
    policy = AutoModelForCausalLM.from_pretrained(resume_from)
else:
    logger.info("Starting training from scratch")
    policy = AutoModelForCausalLM.from_pretrained(MODEL)

ref_model = AutoModelForCausalLM.from_pretrained(MODEL)
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL, num_labels=1)
reward_model.eval()
value_model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)

logger.info("All models loaded")

# --- Initialize Trainer ---
ppo_trainer = PPOTrainer(
    args=cfg,
    processing_class=tok,
    model=policy,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=ds_tokenized,
    eval_dataset=eval_ds_tokenized,
)

# --- Train ---
ppo_trainer.train()

# --- Save final model ---
final_dir = os.path.join(data_dir, f"{MODEL}/ppo-final")
ppo_trainer.save_model(final_dir)
tok.save_pretrained(final_dir)
logger.info(f"Final model saved to {final_dir}")

wandb.finish()
logger.info("Training complete!")
