import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'rl-algos')

files = [f for f in os.listdir(results_dir) if f.endswith('.npy')]
files.sort()

plt.figure(figsize=(10, 6))
plt.rcParams["font.size"] = 16

for fname in files:
    data = np.load(os.path.join(results_dir, fname))
    label = fname.replace('total_reward_', '').replace('.npy', '')
    episodes = np.arange(1, len(data) + 1)
    plt.plot(episodes, data, label=label)

plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward for On-Policy Methods')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'on_policy_results.png'), dpi=300)
plt.show()
