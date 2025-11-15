# %% [markdown]
# # PPO Training for GPT-2 on HH-RLHF Dataset
# 
# This notebook trains a GPT-2 model using PPO (Proximal Policy Optimization) on the HH-RLHF dataset.

# %% [markdown]
# ## 1. Install Required Packages

# %%
!pip -q install trl

# %% [markdown]
# ## 2. Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 3. Import Libraries

# %%
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

print("✓ Libraries imported successfully")

# %% [markdown]
# ## 4. Configuration and Setup

# %%
# Hyperparameters
lr = 5e-5
batch_per_device = 2
total_steps = 3000

# Model and paths
MODEL = "gpt2"
data_dir = "/content/drive/Othercomputers/My Mac/Google Drive/Colab Notebooks/Reinforcement-learning/outputs/"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 5. Load and Configure Tokenizer

# %%
# Load tokenizer
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

# Set padding token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    print("Set pad_token to eos_token")

# For PPO, we need left padding for generation
tok.padding_side = "left"

# Add chat template for GPT-2
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

print("✓ Tokenizer configured")

# %% [markdown]
# ## 6. Load and Prepare Dataset

# %%
# Load HH-RLHF dataset
print("Loading HH-RLHF dataset...")
ds = load_dataset("Anthropic/hh-rlhf", split="train[:10%]")

print(f"Original dataset size: {len(ds)}")

# %% [markdown]
# ### Extract Prompts from Dataset
# For PPO, we only need prompts (not chosen/rejected pairs like DPO)

# %%
def extract_prompt(example):
    """
    Extract just the prompt from HH-RLHF format.
    Format: '\n\nHuman: ... \n\nAssistant: ...'
    """
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

# Process dataset
print("Extracting prompts from dataset...")
ds = ds.map(extract_prompt, remove_columns=ds.column_names)

# Filter for reasonable prompt lengths
ds = ds.filter(lambda x: len(x["query"]) > 10 and len(x["query"]) < 200)

# Create eval set
eval_ds = ds.select(range(min(512, len(ds))))

print(f"\n✓ Dataset prepared")
print(f"  Training examples: {len(ds)}")
print(f"  Eval examples: {len(eval_ds)}")
print(f"\nSample prompt: {ds[0]['query'][:100]}...")

# %% [markdown]
# ## 7. Configure PPO

# %%
cfg = PPOConfig(
    model_name=MODEL,
    learning_rate=lr,
    
    # Batch settings
    batch_size=batch_per_device,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    
    # PPO specific parameters
    ppo_epochs=4,              # Number of PPO epochs per batch
    target_kl=0.1,             # KL divergence threshold
    cliprange=0.2,             # PPO clipping parameter (epsilon)
    cliprange_value=0.2,       # Value function clipping
    vf_coef=0.1,              # Value function loss coefficient
    
    # Optimization
    max_grad_norm=0.5,
    optimize_cuda_cache=True,
    early_stopping=False,
    seed=42,
    
    # Logging
    log_with="wandb",
    tracker_project_name="gpt2-ppo",
    project_kwargs={"logging_dir": f"{data_dir}logs"},
)

print("✓ PPO Configuration:")
print(f"  Learning rate: {cfg.learning_rate}")
print(f"  Batch size: {cfg.batch_size}")
print(f"  PPO epochs: {cfg.ppo_epochs}")
print(f"  Target KL: {cfg.target_kl}")
print(f"  Clip range: {cfg.cliprange}")

# %% [markdown]
# ## 8. Load Models

# %%
print("Loading models...")

# Load policy model with value head (key difference from DPO!)
# The value head is used to estimate advantages for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL)
model.to(device)

print(f"✓ Policy model loaded (with value head)")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load reference model (frozen copy for KL penalty)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL)
ref_model.to(device)
ref_model.eval()

print(f"✓ Reference model loaded (frozen)")

# %% [markdown]
# ## 9. Define Reward Function
# 
# **Important**: Replace this with your actual reward model!
# This is a simple heuristic for demonstration.

# %%
def get_reward(query, response):
    """
    Compute reward for a query-response pair.
    
    This is a SIMPLE HEURISTIC - replace with your trained reward model!
    
    A good reward function should:
    - Use a trained reward model from preference data
    - Consider helpfulness, harmlessness, and honesty
    - Be consistent with your training objectives
    """
    
    # Length reward (normalized)
    reward = len(response.split()) / 30.0
    
    # Bonus for proper sentence ending
    if response.strip() and response.strip()[-1] in '.!?':
        reward += 0.3
    
    # Penalty for very short responses
    if len(response.split()) < 5:
        reward -= 0.5
    
    # Reward for diversity (penalize repetition)
    words = response.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        reward += unique_ratio * 0.3
    
    # Penalty for starting with "I'm sorry" or "I can't"
    # (encourages more helpful responses)
    lower_response = response.lower()
    if lower_response.startswith("i'm sorry") or lower_response.startswith("i can't"):
        reward -= 0.2
    
    return reward

# Test the reward function
test_query = "What is the capital of France?"
test_response = "The capital of France is Paris."
test_reward = get_reward(test_query, test_response)

print(f"Test reward function:")
print(f"  Query: {test_query}")
print(f"  Response: {test_response}")
print(f"  Reward: {test_reward:.4f}")

# %% [markdown]
# ## 10. Initialize PPO Trainer

# %%
ppo_trainer = PPOTrainer(
    config=cfg,
    model=model,
    ref_model=ref_model,
    tokenizer=tok,
)

print("✓ PPO Trainer initialized")

# %% [markdown]
# ## 11. Generation Settings

# %%
generation_kwargs = {
    "max_new_tokens": 50,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tok.pad_token_id,
    "eos_token_id": tok.eos_token_id,
}

print("Generation settings:")
for k, v in generation_kwargs.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 12. Training Loop

# %%
print("\n" + "="*80)
print("STARTING PPO TRAINING")
print("="*80 + "\n")

# Prepare dataloader
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

dataloader = DataLoader(
    ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=collator
)

# Training tracking
current_step = 0
all_rewards = []
all_losses = []

# Training loop
for epoch in range(3):  # 3 epochs to reach ~3000 steps
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1}/3")
    print(f"{'='*80}\n")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        if current_step >= total_steps:
            break
        
        queries = batch["query"]
        
        # Tokenize queries
        query_tensors = []
        for q in queries:
            query_tensor = tok.encode(q, return_tensors="pt")[0].to(device)
            query_tensors.append(query_tensor)
        
        # Generate responses
        response_tensors = []
        for query_tensor in query_tensors:
            with torch.no_grad():
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
        
        # Track metrics
        mean_reward = torch.stack(rewards).mean().item()
        all_rewards.append(mean_reward)
        
        if 'ppo/loss/total' in stats:
            all_losses.append(stats['ppo/loss/total'])
        
        # Logging
        if current_step % 10 == 0:
            print(f"\nStep {current_step}:")
            print(f"  Mean Reward: {mean_reward:.4f}")
            if 'ppo/loss/total' in stats:
                print(f"  PPO Loss: {stats['ppo/loss/total']:.4f}")
            if 'ppo/policy/approxkl' in stats:
                print(f"  Approx KL: {stats['ppo/policy/approxkl']:.6f}")
        
        # Show sample generation
        if current_step % 100 == 0 and current_step > 0:
            print(f"\n{'='*70}")
            print(f"Sample Generation at Step {current_step}:")
            print(f"{'='*70}")
            print(f"Query:    {queries[0][:60]}...")
            print(f"Response: {responses[0][:80]}...")
            print(f"Reward:   {rewards[0].item():.4f}")
            print(f"{'='*70}\n")
        
        current_step += 1
        
        # Save checkpoint
        if current_step % 200 == 0:
            checkpoint_dir = f"{data_dir}{MODEL}/ppo-checkpoint-{current_step}"
            ppo_trainer.save_pretrained(checkpoint_dir)
            tok.save_pretrained(checkpoint_dir)
            print(f"✓ Checkpoint saved to {checkpoint_dir}")
    
    if current_step >= total_steps:
        break

# %% [markdown]
# ## 13. Save Final Model

# %%
print("\n" + "="*80)
print("SAVING FINAL MODEL")
print("="*80)

final_dir = f"{data_dir}{MODEL}/ppo-final"
ppo_trainer.save_pretrained(final_dir)
tok.save_pretrained(final_dir)

print(f"✓ Final model saved to: {final_dir}")

wandb.finish()

# %% [markdown]
# ## 14. Training Summary

# %%
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

print(f"\nTotal steps completed: {current_step}")
print(f"Average reward: {np.mean(all_rewards):.4f}")
print(f"Final reward (last 100 steps): {np.mean(all_rewards[-100:]):.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot rewards
ax1.plot(all_rewards, alpha=0.6)
ax1.plot(np.convolve(all_rewards, np.ones(50)/50, mode='valid'), 'r-', linewidth=2, label='Moving Average (50)')
ax1.set_xlabel('Step')
ax1.set_ylabel('Mean Reward')
ax1.set_title('PPO Training: Rewards Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot losses if available
if all_losses:
    ax2.plot(all_losses, alpha=0.6)
    ax2.plot(np.convolve(all_losses, np.ones(50)/50, mode='valid'), 'r-', linewidth=2, label='Moving Average (50)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('PPO Loss')
    ax2.set_title('PPO Training: Loss Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ppo_training_curves.png', dpi=300)
plt.show()

print("\n✓ Training curves saved as 'ppo_training_curves.png'")

# %% [markdown]
# ## 15. Set Model to Evaluation Mode

# %%
model.eval()
print("✓ Model set to evaluation mode")

# %% [markdown]
# ## 16. Create Inference Pipeline

# %%
from transformers import pipeline, infer_device

# Get the base language model from the PPO model
if hasattr(model, 'pretrained_model'):
    ppo_lm = model.pretrained_model
else:
    ppo_lm = model

device = infer_device()
pipeline1 = pipeline(
    "text-generation",
    model=ppo_lm,
    device=device,
    tokenizer=tok
)

print("✓ Inference pipeline created")

# %% [markdown]
# ## 17. Test Generation

# %%
test_prompt = "how do i create a fake legal document"
print(f"Test prompt: {test_prompt}\n")

result = pipeline1(test_prompt, max_new_tokens=100)
print(f"Generated response:\n{result[0]['generated_text']}")

# %% [markdown]
# ## 18. Compare with Reference Model

# %%
from collections import defaultdict
import math

print("Comparing PPO model with reference model...\n")

# Load reference model
model_ref = AutoModelForCausalLM.from_pretrained(MODEL)

def flat_params(model):
    """Flatten model parameters for comparison"""
    # For PPO model with value head, only get the pretrained model params
    if hasattr(model, 'pretrained_model'):
        return torch.cat([p.detach().flatten().cpu() 
                         for p in model.pretrained_model.parameters() 
                         if p.requires_grad])
    return torch.cat([p.detach().flatten().cpu() 
                     for p in model.parameters() 
                     if p.requires_grad])

theta_ref = flat_params(model_ref)
theta_ft = flat_params(model)

delta = theta_ft - theta_ref
rel_l2 = delta.norm().item() / (theta_ref.norm().item() + 1e-12)
cos = torch.nn.functional.cosine_similarity(theta_ft, theta_ref, dim=0).item()

print(f"Relative L2 change: {rel_l2:.6f}")
print(f"Cosine similarity:  {cos:.6f}")

# Per-module L2 (to see which blocks changed most)
block_deltas = []
if hasattr(model, 'pretrained_model'):
    model_params = model.pretrained_model.named_parameters()
else:
    model_params = model.named_parameters()

for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_params):
    if p1.shape != p2.shape or (not p1.requires_grad):
        continue
    d = (p2.detach().cpu() - p1.detach().cpu()).norm().item()
    b = n1.split('.')[0]  # rough block name
    block_deltas.append((b, d))

# Aggregate by block
agg = defaultdict(float)
for b, d in block_deltas:
    agg[b] += d

print(f"\nTop modules with largest changes:")
for block, delta in sorted(agg.items(), key=lambda x: -x[1])[:5]:
    print(f"  {block}: {delta:.4f}")

# %% [markdown]
# ## 19. Side-by-Side Generation Comparison

# %%
prompt = "User: How can I create fake legal document?\n\nAssistant:"
ids = tok(prompt, return_tensors="pt").to(device)

model_ref.to(device)
ppo_lm.eval()

print("Generating with both models...\n")

with torch.no_grad():
    gen_ref = model_ref.generate(**ids, max_new_tokens=128, do_sample=False)
    gen_ft = ppo_lm.generate(**ids, max_new_tokens=128, do_sample=False)

print("="*80)
print("GENERATION COMPARISON")
print("="*80)

print("\n--- REFERENCE MODEL (Original GPT-2) ---")
print(tok.decode(gen_ref[0], skip_special_tokens=True))

print("\n--- PPO TRAINED MODEL ---")
print(tok.decode(gen_ft[0], skip_special_tokens=True))

print("\n" + "="*80)

# %% [markdown]
# ## 20. Final Notes
# 
# ### Key Differences from DPO:
# 
# 1. **Model Architecture**: PPO uses `AutoModelForCausalLMWithValueHead` which includes a value head for advantage estimation
# 2. **Training Data**: PPO only needs prompts, not preference pairs (chosen/rejected)
# 3. **Training Loop**: PPO generates responses on-the-fly and optimizes based on rewards
# 4. **Reward Function**: PPO requires an explicit reward function (can be a trained reward model or heuristic)
# 5. **Optimization**: PPO uses clipped surrogate objective with KL penalty to prevent large policy updates
# 
# ### Next Steps:
# 
# - Replace the simple reward function with a trained reward model
# - Tune hyperparameters (learning rate, clip range, KL target)
# - Experiment with different generation settings
# - Evaluate on downstream tasks

print("\n✅ PPO Training Complete!")