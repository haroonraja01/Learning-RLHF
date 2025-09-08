Reinforcement Learning for Model Fine-tuning
===============================
In this document we provide a brief introduction to reinforcement learning (RL). In the following we first descibe the problem setup and the challenges while solving, then we will will briefly describe the policy gradient style algorithms and why such algorithms are preferred for solving RL in the context of large language models (LLM), and finally we provide experimental results on classical cartpole environment and fine-tuning GPT-2 using RLHF.
# Introduction



# On- vs Off-Policy: Key Take-aways

## Why On-policy?

**Off-policy (Q-learning, DQN, SAC)**

✅ More sample-efficient (reuse replay buffer).

⚠️ Can be harder to stabilize (distribution shift, divergence).

⚖️ Not inherently more “computationally expensive” — often cheaper per gradient step.

**On-policy (REINFORCE, A2C, PPO)**

✅ Simple & stable (fresh data, unbiased gradients).

❌ Less sample-efficient (must discard old trajectories).

🚀 Easier to scale to large policies (why PPO is used in RLHF).

## Algorithm lineage:

- REINFORCE → simplest on-policy policy gradient; high variance.
- Actor–Critic → adds a value function to reduce variance & improve learning.
- PPO / TRPO → clip or constrain updates for stability at scale.

# Reading list for basic understanding

- 📖 Sutton & Barto (2018, 2nd ed.) — Ch. 13: Policy Gradient Methods
Foundations: REINFORCE, actor–critic, compatible function approximation, natural gradients.
- 📄 Mnih et al., 2016 — “Asynchronous Methods for Deep Reinforcement Learning (A3C)”
Practical actor–critic with entropy regularization and distributed rollouts.
- 📄 Schulman et al., 2017 — “Proximal Policy Optimization Algorithms (PPO)”
Stable, scalable on-policy method; the backbone of RLHF training.
- 📄 Christiano et al., 2017 — “Deep Reinforcement Learning from Human Preferences”
Reward modeling from human comparisons; first demonstration of RL with human preference signals.
- 📄 Ziegler et al., 2019 — “Fine-Tuning Language Models from Human Preferences”
Full RLHF pipeline on language models (SFT → reward model → PPO fine-tuning).