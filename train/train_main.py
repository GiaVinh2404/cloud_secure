# train_main.py (checkpoint theo reward action + predictor)

import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from env.cloud_security_env import CloudSecurityEnv
from agents.base_dqn_agent import BaseDQNAgent

CHECKPOINT_DIR = "checkpoints"
RESULTS_CSV = "train_results.csv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== Visualization helper =====
def visualize_history(reward_history, episode):
    episodes = np.arange(1, len(reward_history) + 1)

    # Tính reward từng agent
    attacker_rewards = [r['attacker_agent'] for r in reward_history]
    predictor_rewards = [r['predictor_agent'] for r in reward_history]
    action_rewards = [r['action_agent'] for r in reward_history]
    predictor_action_total = [p + a for p, a in zip(predictor_rewards, action_rewards)]

    plt.figure(figsize=(14, 10))

    # --- Biểu đồ 1: Attacker ---
    plt.subplot(2, 2, 1)
    plt.plot(episodes, attacker_rewards, color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Attacker Agent Reward per Episode")
    plt.grid(True, linestyle="--", alpha=0.5)

    # --- Biểu đồ 2: Predictor ---
    plt.subplot(2, 2, 2)
    plt.plot(episodes, predictor_rewards, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Predictor Agent Reward per Episode")
    plt.grid(True, linestyle="--", alpha=0.5)

    # --- Biểu đồ 3: Action ---
    plt.subplot(2, 2, 3)
    plt.plot(episodes, action_rewards, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Action Agent Reward per Episode")
    plt.grid(True, linestyle="--", alpha=0.5)

    # --- Biểu đồ 4: Predictor + Action ---
    plt.subplot(2, 2, 4)
    plt.plot(episodes, predictor_action_total, color="green")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Predictor + Action Total Reward per Episode")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

# ===== CSV Logger =====
def init_csv_logger(path):
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "AttackerReward", "PredictorReward", "ActionReward", "TotalReward",
                         "AttackerEps", "PredictorEps", "ActionEps"])

def log_to_csv(path, episode, att_r, pred_r, act_r, total_r, eps_att, eps_pred, eps_act):
    with open(path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, att_r, pred_r, act_r, total_r, eps_att, eps_pred, eps_act])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init env & agents
    env = CloudSecurityEnv(max_steps=500)
    agents = {
        "attacker_agent": BaseDQNAgent(env.action_spaces["attacker_agent"], env.observation_spaces["attacker_agent"].shape[0], device=device),
        "predictor_agent": BaseDQNAgent(env.action_spaces["predictor_agent"], env.observation_spaces["predictor_agent"].shape[0], device=device),
        "action_agent": BaseDQNAgent(env.action_spaces["action_agent"], env.observation_spaces["action_agent"].shape[0], device=device)
    }

    # Checkpoint paths
    ckpt_paths = {name: os.path.join(CHECKPOINT_DIR, f"{name}_best.pth") for name in agents}

    # CSV log init
    init_csv_logger(RESULTS_CSV)

    # Training params
    num_episodes = 200
    best_total_reward_PA = -float("inf")  # best predictor + action reward
    reward_history = []

    for episode in range(1, num_episodes+1):
        start_time = time.time()
        env.reset()

        ep_rewards = {name: 0.0 for name in agents}
        last_obs, last_action = {}, {}

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc

            action = agents[agent].act(obs)
            env.step(action)

            ep_rewards[agent] += reward

            # Reward shaping cho attacker
            if agent == "action_agent":
                label = env.current_label
                if label > 0 and action == 0:  # attack success
                    ep_rewards["attacker_agent"] += 3.0
                elif label > 0 and action == 1:  # blocked
                    ep_rewards["attacker_agent"] -= 1.0
                elif label == 0 and action == 1:  # false positive
                    ep_rewards["attacker_agent"] += 0.5
                elif label == 0 and action == 0:  # benign allowed
                    ep_rewards["attacker_agent"] += 0.2

            # Replay memory update
            if agent in last_obs:
                agents[agent].remember(last_obs[agent], last_action[agent], reward, obs, done)
                agents[agent].replay(64)

            last_obs[agent] = obs
            last_action[agent] = action

        total_reward = sum(ep_rewards.values())
        total_reward_PA = ep_rewards["predictor_agent"] + ep_rewards["action_agent"]

        # Save CSV log
        log_to_csv(RESULTS_CSV, episode, ep_rewards["attacker_agent"], ep_rewards["predictor_agent"], ep_rewards["action_agent"],
                   total_reward, agents["attacker_agent"].epsilon, agents["predictor_agent"].epsilon, agents["action_agent"].epsilon)

        # Track history
        reward_history.append(ep_rewards)

        # Save best checkpoint based on Predictor + Action reward
        if total_reward_PA > best_total_reward_PA:
            best_total_reward_PA = total_reward_PA
            agents["predictor_agent"].save(ckpt_paths["predictor_agent"])
            agents["action_agent"].save(ckpt_paths["action_agent"])
            print(f"[EP {episode}] New best Predictor+Action reward {best_total_reward_PA:.2f}, saved checkpoints.")

        # Update target networks every 5 episodes
        if episode % 5 == 0:
            for ag in agents.values():
                ag.update_target()

        # Visualization every 20 episodes
        if episode % 20 == 0:
            visualize_history(reward_history, episode)

        elapsed = time.time() - start_time
        print(f"[EP {episode}] R_att={ep_rewards['attacker_agent']:.2f}, R_pred={ep_rewards['predictor_agent']:.2f}, R_act={ep_rewards['action_agent']:.2f}, Total={total_reward:.2f}, Time={elapsed:.1f}s")

    print("Training finished. Results saved to", RESULTS_CSV)
