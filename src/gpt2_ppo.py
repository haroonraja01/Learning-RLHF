# Cell 1: Install packages
!pip -q install trl

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Import libraries
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb

# Cell 4: Main Training Code
lr = 5e-5
batch_per_device = 2

MODEL = "gpt2"
data_dir = "/content/drive/Othercomputers/My Mac/Google Drive/Colab Notebooks/Reinforcement-learning/outputs/"

# --- Tokenizer (GPT-2 has no pad token) ---
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# For PPO, we need left padding for generation
tok.padding_side = "left"

if MODEL == 'gpt2':
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

# Load dataset - for PPO we only need prompts
ds = load_dataset("Anthropic/hh-rlhf", split="train[:10%]")

# Extract prompts from the dataset
def extract_prompt(example):
    """Extract just the prompt from HH-RLHF format"""
    text = example["chosen"]
    # Split by Assistant responses
    parts = text.split("\n\nAssistant:")
    if len(parts) > 0:
        # Get everything before the last Assistant response
        human_part = parts[-2] if len(parts) > 1 else parts[0]
        if "\n\nHuman:" in human_part:
            prompt = human_part.split("\n\nHuman:")[-1].strip()
        else:
            prompt = human_part.strip()
        return {"query": prompt}
    return {"query": text.strip()}

# Process dataset to extract only prompts
ds = ds.map(extract_prompt, remove_columns=ds.column_names)
ds = ds.filter(lambda x: len(x["query"]) > 10 and len(x["query"]) < 200)

eval_ds = ds.select(range(min(512, len(ds))))

print(f"Dataset size: {len(ds)}")
print(f"Sample prompt: {ds[0]['query']}")

# --- PPO Config ---
cfg = PPOConfig(
    model_name=MODEL,
    learning_rate=lr,
    batch_size=batch_per_device,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    ppo_epochs=4,
    
    # PPO specific parameters
    target_kl=0.1,           # KL divergence threshold
    cliprange=0.2,           # PPO clipping parameter
    cliprange_value=0.2,     # Value function clipping
    vf_coef=0.1,            # Value function coefficient
    
    # Training parameters
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,
    seed=42,
    
    # Logging
    log_with="wandb",
    tracker_project_name="gpt2-ppo",
    project_kwargs={"logging_dir": f"{data_dir}/logs"},
)

# --- Load Model with Value Head (key difference from DPO!) ---
# PPO requires a model with a value head for advantage estimation
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL)

# Load reference model (frozen copy for KL penalty)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL)

# --- Define Reward Function ---
def get_reward(query, response):
    """
    Reward function - replace this with your actual reward model
    For now, using a simple heuristic
    """
    # Simple reward: prefer longer, coherent responses
    reward = len(response.split()) / 30.0
    
    # Bonus for proper ending
    if response.strip() and response.strip()[-1] in '.!?':
        reward += 0.3
    
    # Penalty for very short responses
    if len(response.split()) < 5:
        reward -= 0.5
    
    # Penalty for repetition
    words = response.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        reward += unique_ratio * 0.3
    
    return reward

# --- Initialize PPO Trainer ---
ppo_trainer = PPOTrainer(
    config=cfg,
    model=model,
    ref_model=ref_model,
    tokenizer=tok,
)

# --- Generation Settings ---
generation_kwargs = {
    "max_new_tokens": 50,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tok.pad_token_id,
    "eos_token_id": tok.eos_token_id,
}

# --- Training Loop ---
print("\n" + "="*80)
print("STARTING PPO TRAINING")
print("="*80 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ref_model.to(device)

total_steps = 3000
current_step = 0

# Create dataloader
from torch.utils.data import DataLoader

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

dataloader = DataLoader(
    ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=collator
)

for epoch in range(3):  # 3 epochs to match ~3000 steps
    for batch_idx, batch in enumerate(dataloader):
        if current_step >= total_steps:
            break
        
        queries = batch["query"]
        
        # Tokenize queries
        query_tensors = [tok.encode(q, return_tensors="pt")[0].to(device) for q in queries]
        
        # Generate responses
        response_tensors = []
        for query_tensor in query_tensors:
            response = ppo_trainer.generate(
                query_tensor.unsqueeze(0),
                **generation_kwargs
            )
            # Extract only the generated part (not the query)
            response_tensor = response[0][len(query_tensor):]
            response_tensors.append(response_tensor)
        
        # Decode responses
        responses = [tok.decode(r, skip_special_tokens=True) for r in response_tensors]
        
        # Compute rewards
        rewards = []
        for query, response in zip(queries, responses):
            reward = get_reward(query, response)
            rewards.append(torch.tensor(reward, device=device))
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log statistics
        if current_step % 10 == 0:
            mean_reward = torch.stack(rewards).mean().item()
            print(f"Step {current_step}: Mean Reward = {mean_reward:.4f}")
            
            if current_step % 100 == 0 and current_step > 0:
                print(f"\nSample Generation:")
                print(f"  Query: {queries[0][:60]}...")
                print(f"  Response: {responses[0][:80]}...")
                print(f"  Reward: {rewards[0].item():.4f}\n")
        
        current_step += 1
        
        # Save checkpoint
        if current_step % 200 == 0:
            checkpoint_dir = f"{data_dir}{MODEL}/ppo-checkpoint-{current_step}"
            ppo_trainer.save_pretrained(checkpoint_dir)
            tok.save_pretrained(checkpoint_dir)
            print(f"✓ Checkpoint saved to {checkpoint_dir}")

# Save final model
final_dir = f"{data_dir}{MODEL}/ppo-final"
ppo_trainer.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
print(f"\n✓ Final model saved to {final_dir}")

wandb.finish()

# Cell 5: Test the model
model.eval()

# Cell 6: Create pipeline for inference
from transformers import pipeline, infer_device

device = infer_device()
pipeline1 = pipeline("text-generation", model=model.pretrained_model, device=device, tokenizer=tok)

# Cell 7: Test generation
pipeline1("how do i create a fake legal document")

# Cell 8: Compare with reference model
from collections import defaultdict
import math

model_ref = AutoModelForCausalLM.from_pretrained(MODEL)

def flat_params(model):
    # For PPO model with value head, only get the pretrained model params
    if hasattr(model, 'pretrained_model'):
        return torch.cat([p.detach().flatten().cpu() for p in model.pretrained_model.parameters() if p.requires_grad])
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad])

theta_ref = flat_params(model_ref)
theta_ft = flat_params(model)

delta = theta_ft - theta_ref
rel_l2 = delta.norm().item() / (theta_ref.norm().item() + 1e-12)
cos = torch.nn.functional.cosine_similarity(theta_ft, theta_ref, dim=0).item()
print(f"Relative L2 change: {rel_l2:.6f}")
print(f"Cosine similarity:  {cos:.6f}")

# Cell 9: Compare generations
prompt = "User: How can I create fake legal document?\n\nAssistant:"
ids = tok(prompt, return_tensors="pt").to(device)

# Get the base language model from the PPO model
if hasattr(model, 'pretrained_model'):
    ppo_lm = model.pretrained_model
else:
    ppo_lm = model

model_ref.to(device)
ppo_lm.eval()

gen_ref = model_ref.generate(**ids, max_new_tokens=128, do_sample=False)
gen_ft = ppo_lm.generate(**ids, max_new_tokens=128, do_sample=False)

print("\n--- REF ---\n", tok.decode(gen_ref[0], skip_special_tokens=True))
print("\n--- PPO ---\n", tok.decode(gen_ft[0], skip_special_tokens=True))