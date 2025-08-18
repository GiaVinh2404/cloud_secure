import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from env.cloud_security_env import CloudSecurityEnv
from agents.base_dqn_agent import BaseDQNAgent
from config.settings import DQN_CONFIG

CHECKPOINT_DIR = "checkpoints"
RESULTS_CSV = "train_results.csv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== Visualization helper =====
def visualize_history(reward_history):
    episodes = np.arange(1, len(reward_history) + 1)
    predictor_rewards = [r['predictor_agent'] for r in reward_history]
    action_rewards = [r['action_agent'] for r in reward_history]
    total_rewards = [p + a for p, a in zip(predictor_rewards, action_rewards)]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(episodes, predictor_rewards, color="blue")
    plt.title("Predictor Reward"); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(episodes, action_rewards, color="orange")
    plt.title("Action Reward"); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(episodes, total_rewards, color="green")
    plt.title("Total Reward"); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)

    plt.tight_layout()
    plt.show()

# ===== CSV Logger =====
def init_csv_logger(path):
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "PredictorReward", "ActionReward", "TotalReward",
                             "PredictorEps", "ActionEps"])

def log_to_csv(path, episode, pred_r, act_r, total_r, eps_pred, eps_act):
    with open(path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, pred_r, act_r, total_r, eps_pred, eps_act])


# ===== Training Loop =====
def train(num_episodes=DQN_CONFIG["num_episodes"], seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    env = CloudSecurityEnv(
        dataset_path="data/processed/cic2018_processed.parquet",
        max_steps=500
    )

    agents = {
        "predictor_agent": BaseDQNAgent(env.action_spaces["predictor_agent"], env.observation_spaces["predictor_agent"].shape[0], device=device),
        "action_agent": BaseDQNAgent(env.action_spaces["action_agent"], env.observation_spaces["action_agent"].shape[0], device=device)
    }

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_combined_checkpoint.pth")
    best_total_reward = -float("inf")

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        agents["predictor_agent"].load_state_dict(checkpoint["predictor_agent"])
        agents["action_agent"].load_state_dict(checkpoint["action_agent"])

        best_total_reward = checkpoint["best_total_reward_PA"]
        print(f"[INFO] Loaded checkpoint with best total reward: {best_total_reward:.2f}")

    init_csv_logger(RESULTS_CSV)
    reward_history = []

    for episode in range(1, num_episodes + 1):
        print(f"\n=== EPISODE {episode}/{num_episodes} START ===")
        start_time = time.time()

        env.reset()
        last_obs = {agent: env.observe(agent) for agent in agents}
        last_action, last_reward = {}, {agent: 0.0 for agent in agents}
        ep_rewards = {agent: 0.0 for agent in agents}

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc

            if agent not in agents:
                env.step(0)
                continue

            ep_rewards[agent] += reward - last_reward[agent]
            last_reward[agent] = reward

            action = agents[agent].act(obs)
            env.step(action)

            if agent in last_obs and agent in last_action:
                agents[agent].remember(last_obs[agent], last_action[agent], reward, obs, done)
                agents[agent].replay()

            last_obs[agent], last_action[agent] = obs, action

        total_reward = ep_rewards["predictor_agent"] + ep_rewards["action_agent"]
        reward_history.append(ep_rewards.copy())

        # Logging
        log_to_csv(RESULTS_CSV, episode,
                   ep_rewards["predictor_agent"],
                   ep_rewards["action_agent"],
                   total_reward,
                   agents["predictor_agent"].epsilon,
                   agents["action_agent"].epsilon)

        WARM_UP_EPISODES = 50
        # Save best checkpoint
        if episode > WARM_UP_EPISODES and total_reward > best_total_reward:
            # Điều kiện mới: Chỉ lưu checkpoint nếu epsilon đã thấp hơn ngưỡng
            if agents["predictor_agent"].epsilon < 0.2 and agents["action_agent"].epsilon < 0.2:
                best_total_reward = total_reward
                checkpoint_data = {
                    "predictor_agent": agents["predictor_agent"].get_state(),
                    "action_agent": agents["action_agent"].get_state(),
                    "best_total_reward_PA": best_total_reward,
                }
                torch.save(checkpoint_data, checkpoint_path)
                print(f"[EP {episode}] ✅ New best total reward {best_total_reward:.2f}, checkpoint saved.")
            else:
                print(f"[EP {episode}] ⚠️ High reward but epsilon is too high ({agents['predictor_agent'].epsilon:.2f}). Not saving checkpoint.")

        # Periodic target update
        if episode % 5 == 0:
            for ag in agents.values():
                ag.update_target()

        # Print summary
        elapsed = time.time() - start_time
        avg_last_10 = np.mean([r["predictor_agent"] + r["action_agent"] for r in reward_history[-10:]])
        print(f"Predictor Reward: {ep_rewards['predictor_agent']:.2f}\n"
              f"Action Reward: {ep_rewards['action_agent']:.2f}\n"
              f"Total: {total_reward:.2f}\n"
              f"10-ep Avg: {avg_last_10:.2f}\n"
              f"Time: {elapsed:.1f}s")

        for ag in agents.values():
            ag.decay_epsilon()
        # Optional: visualize
        # if episode % 20 == 0: visualize_history(reward_history)

    print("✅ Training finished. Results saved to", RESULTS_CSV)


if __name__ == "__main__":
    train(num_episodes=DQN_CONFIG["num_episodes"])