import os
import time
import matplotlib.pyplot as plt
import torch
from env.cloud_security_env import CloudSecurityEnv
from agents.base_dqn_agent import BaseDQNAgent

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def visualize_history(predictor_history, action_history, reward_history, episode):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([sum(ep) for ep in predictor_history], label='Predictor Agent', color='blue')
    plt.plot([sum(ep) for ep in action_history], label='Action Agent', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Sum of Actions')
    plt.title(f'Action Summary up to Episode {episode+1}')
    plt.legend(loc="upper right")
    plt.subplot(1, 2, 2)
    plt.plot([sum(r) for r in reward_history], label='Total Reward', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Progress')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = CloudSecurityEnv(max_steps=1000)
    print("Environment initialized.")

    predictor_agent = BaseDQNAgent(
        action_space=env.action_spaces['predictor_agent'],
        obs_size=env.observation_spaces['predictor_agent'].shape[0],
        device=device
    )
    print("Predictor agent initialized.")

    action_agent = BaseDQNAgent(
        action_space=env.action_spaces['action_agent'],
        obs_size=env.observation_spaces['action_agent'].shape[0],
        device=device
    )
    print("Action agent initialized.")

    predictor_ckpt_path = os.path.join(CHECKPOINT_DIR, "predictor_agent_best.pth")
    action_ckpt_path = os.path.join(CHECKPOINT_DIR, "action_agent_best.pth")

    print(f"Predictor checkpoint path: {predictor_ckpt_path}")
    print(f"Action checkpoint path: {action_ckpt_path}")

    if os.path.exists(predictor_ckpt_path):
        predictor_agent.load(predictor_ckpt_path)
        print("Loaded predictor agent from checkpoint.")
    else:
        print("Predictor agent checkpoint not found.")

    if os.path.exists(action_ckpt_path):
        action_agent.load(action_ckpt_path)
        print("Loaded action agent from checkpoint.")
    else:
        print("Action agent checkpoint not found.")

    num_episodes = 100  # Tăng số episode để agent học tốt hơn
    predictor_history = []
    action_history = []
    reward_history = []

    best_total_reward = None
    best_episode = None

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1} ===")
        start_time = time.time()
        episode_predictor = []
        episode_action = []

        correct_predictor = 0
        total_predictor = 0
        correct_action = 0
        total_action = 0

        env.reset()
        last_obs = {}
        last_action = {}
        step_count = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if agent == 'predictor_agent':
                action = predictor_agent.act(observation)
                episode_predictor.append(action)
                label = env.labels[env.idx] if env.idx < len(env.labels) else 0
                if action == label:
                    correct_predictor += 1
                total_predictor += 1
            else:
                action = action_agent.act(observation)
                episode_action.append(action)
                pred = env.last_prediction
                if (pred == 1 and action == 1) or (pred == 0 and action == 0):
                    correct_action += 1
                total_action += 1

            if agent in last_obs:
                if agent == 'predictor_agent':
                    predictor_agent.remember(last_obs[agent], last_action[agent], reward, observation, done)
                else:
                    action_agent.remember(last_obs[agent], last_action[agent], reward, observation, done)

            last_obs[agent] = observation
            last_action[agent] = action

            env.step(action)
            step_count += 1

            # Replay ngay sau mỗi bước để cập nhật nhanh hơn
            if agent == 'predictor_agent':
                predictor_agent.replay(64)
            else:
                action_agent.replay(64)

        predictor_acc = correct_predictor / total_predictor if total_predictor > 0 else 0
        action_acc = correct_action / total_action if total_action > 0 else 0

        print(f"Episode {episode+1} finished. Steps: {step_count}")
        print(f"Predictor accuracy: {predictor_acc:.4f} ({correct_predictor}/{total_predictor})")
        print(f"Action accuracy: {action_acc:.4f} ({correct_action}/{total_action})")

        # Cập nhật target network thường xuyên hơn
        if (episode + 1) % 5 == 0:
            predictor_agent.update_target()
            action_agent.update_target()
            print("Target networks updated.")

        rewards = env.rewards
        print(f"Episode {episode+1}: Rewards: {rewards}")

        predictor_history.append(episode_predictor)
        action_history.append(episode_action)
        reward_history.append(list(rewards.values()))

        total_reward = sum(rewards.values())
        if best_total_reward is None or total_reward > best_total_reward:
            best_total_reward = total_reward
            best_episode = episode + 1
            predictor_agent.save(predictor_ckpt_path)
            action_agent.save(action_ckpt_path)
            print(f"Saved best checkpoint at episode {best_episode} with total reward {best_total_reward}.")

        if (episode + 1) % 10 == 0:
            visualize_history(predictor_history, action_history, reward_history, episode)
            print(f"Best checkpoint so far: episode {best_episode} with total reward {best_total_reward}")

        elapsed = time.time() - start_time
        print(f"Episode {episode+1} time: {elapsed:.2f} seconds")