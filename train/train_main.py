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
def visualize_history(reward_history):
    episodes = np.arange(1, len(reward_history) + 1)
    predictor_rewards = [r['predictor_agent'] for r in reward_history]
    action_rewards = [r['action_agent'] for r in reward_history]
    predictor_action_total = [p + a for p, a in zip(predictor_rewards, action_rewards)]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(episodes, predictor_rewards, color="blue")
    plt.title("Predictor Reward per Episode")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(episodes, action_rewards, color="orange")
    plt.title("Action Reward per Episode")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(episodes, predictor_action_total, color="green")
    plt.title("Predictor + Action Total Reward")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)

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

# ===== Training =====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = CloudSecurityEnv(
        dataset_path="data/processed/cic2018_processed.parquet",
        max_steps=500
    )

    agents = {
        "predictor_agent": BaseDQNAgent(env.action_spaces["predictor_agent"], env.observation_spaces["predictor_agent"].shape[0], device=device),
        "action_agent": BaseDQNAgent(env.action_spaces["action_agent"], env.observation_spaces["action_agent"].shape[0], device=device)
    }

    # Use a single, shared checkpoint to store all agent states and metadata
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_combined_checkpoint.pth")
    best_total_reward_PA = -float("inf")

    # ===== Load checkpoint =====
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agents["predictor_agent"].load_state_dict(checkpoint["predictor_agent"])
        agents["action_agent"].load_state_dict(checkpoint["action_agent"])
        best_total_reward_PA = checkpoint["best_total_reward_PA"]
        print(f"[INFO] Loaded combined checkpoint from {checkpoint_path} with best total reward: {best_total_reward_PA:.2f}")

    # ===== CSV =====
    init_csv_logger(RESULTS_CSV)

    num_episodes = 200
    reward_history = []

    for episode in range(1, num_episodes + 1):
        print(f"\n=== EPISODE {episode} START ===")
        start_time = time.time()
        env.reset()

        last_obs = {agent: env.observe(agent) for agent in agents}
        last_action = {}
        ep_rewards = {agent: 0.0 for agent in agents}
        last_reward = {agent: 0.0 for agent in agents}

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
                agents[agent].replay(64)

            last_obs[agent] = obs
            last_action[agent] = action

        total_reward_PA = ep_rewards["predictor_agent"] + ep_rewards["action_agent"]

        log_to_csv(RESULTS_CSV, episode,
                   ep_rewards["predictor_agent"],
                   ep_rewards["action_agent"],
                   total_reward_PA,
                   agents["predictor_agent"].epsilon,
                   agents["action_agent"].epsilon)

        reward_history.append(ep_rewards.copy())

        # Save best checkpoint
        if total_reward_PA > best_total_reward_PA:
            best_total_reward_PA = total_reward_PA
            checkpoint_data = {
                "predictor_agent": agents["predictor_agent"].get_state(),
                "action_agent": agents["action_agent"].get_state(),
                "best_total_reward_PA": best_total_reward_PA,
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"[EP {episode}] New best Predictor+Action reward {best_total_reward_PA:.2f}, checkpoints saved.")

        # Update target networks every 5 episodes
        if episode % 5 == 0:
            for ag in agents.values():
                ag.update_target()

        # # Visualization every 20 episodes
        # if episode % 20 == 0:
        #     visualize_history(reward_history)

        elapsed = time.time() - start_time
        print(f"Predictor Reward: {ep_rewards['predictor_agent']:.2f}")
        print(f"Action Reward:    {ep_rewards['action_agent']:.2f}")
        print(f"Total Reward:     {total_reward_PA:.2f}")
        print(f"Time: {elapsed:.1f}s\n")

    print("Training finished. Results saved to", RESULTS_CSV)