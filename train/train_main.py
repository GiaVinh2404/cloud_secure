import os
import time
import matplotlib.pyplot as plt
import torch
from env.cloud_security_env import CloudSecurityEnv
from agents.base_dqn_agent import BaseDQNAgent

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def visualize_history(predictor_history, action_history, attacker_history, reward_history, episode):
    plt.figure(figsize=(18, 5))

    # Tổng số actions theo episode của từng agent
    plt.subplot(1, 3, 1)
    plt.plot([sum(ep) for ep in predictor_history], label='Predictor Agent', color='blue')
    plt.plot([sum(ep) for ep in action_history], label='Action Agent', color='orange')
    plt.plot([sum(ep) for ep in attacker_history], label='Attacker Agent', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Sum of Actions')
    plt.title(f'Action Summary up to Episode {episode+1}')
    plt.legend(loc="upper right")

    # Tổng reward toàn môi trường theo episode
    plt.subplot(1, 3, 2)
    total_rewards = [r.get('predictor_agent', 0) + r.get('action_agent', 0) for r in reward_history]
    plt.plot(total_rewards, label='Total Reward (Predictor+Action)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Progress')
    plt.legend(loc="upper right")

    # Reward theo từng agent riêng biệt
    plt.subplot(1, 3, 3)
    plt.plot(range(len(reward_history)), [r.get('predictor_agent', 0) for r in reward_history], label='Reward Predictor', color='blue')
    plt.plot(range(len(reward_history)), [r.get('action_agent', 0) for r in reward_history], label='Reward Action', color='orange')
    plt.plot(range(len(reward_history)), [r.get('attacker_agent', 0) for r in reward_history], label='Reward Attacker', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward per Agent')
    plt.title('Agent Rewards')
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = CloudSecurityEnv(max_steps=1000)
    print("Environment initialized.")

    attacker_agent = BaseDQNAgent(
        action_space=env.action_spaces['attacker_agent'],
        obs_size=env.observation_spaces['attacker_agent'].shape[0],
        device=device
    )
    print("Attacker agent initialized.")

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

    attacker_ckpt_path = os.path.join(CHECKPOINT_DIR, "attacker_agent_best.pth")
    predictor_ckpt_path = os.path.join(CHECKPOINT_DIR, "predictor_agent_best.pth")
    action_ckpt_path = os.path.join(CHECKPOINT_DIR, "action_agent_best.pth")

    if os.path.exists(attacker_ckpt_path):
        attacker_agent.load(attacker_ckpt_path)
        print("Loaded attacker agent from checkpoint.")
    else:
        print("Attacker agent checkpoint not found.")

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

    num_episodes = 100
    predictor_history = []
    action_history = []
    attacker_history = []
    reward_history = []

    best_total_reward = None
    best_episode = None

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1} ===")
        start_time = time.time()

        episode_predictor = []
        episode_action = []
        episode_attacker = []

        correct_predictor = 0
        total_predictor = 0
        correct_action = 0
        total_action = 0

        env.reset()
        last_obs = {}
        last_action = {}
        step_count = 0

        # Cộng dồn reward cho từng agent trong episode
        episode_reward_predictor = 0
        episode_reward_action = 0
        episode_reward_attacker = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if agent == 'predictor_agent':
                action = predictor_agent.act(observation)
                episode_predictor.append(action)
                label = getattr(env, 'current_label', 0)
                if action == label:
                    correct_predictor += 1
                total_predictor += 1
                episode_reward_predictor += reward
            elif agent == 'action_agent':
                action = action_agent.act(observation)
                episode_action.append(action)
                pred = getattr(env, 'last_prediction', 0)
                if (pred == 1 and action == 1) or (pred == 0 and action == 0):
                    correct_action += 1
                total_action += 1
                episode_reward_action += reward
            elif agent == 'attacker_agent':
                action = attacker_agent.act(observation)
                episode_attacker.append(action)
                episode_reward_attacker += reward

            # Ghi nhớ trạng thái trước, để replay
            if agent in last_obs:
                if agent == 'predictor_agent':
                    predictor_agent.remember(last_obs[agent], last_action[agent], reward, observation, done)
                elif agent == 'action_agent':
                    action_agent.remember(last_obs[agent], last_action[agent], reward, observation, done)
                elif agent == 'attacker_agent':
                    attacker_agent.remember(last_obs[agent], last_action[agent], reward, observation, done)

            last_obs[agent] = observation
            last_action[agent] = action

            env.step(action)
            step_count += 1

            # Replay ngay sau mỗi bước cho agent tương ứng
            if agent == 'predictor_agent':
                predictor_agent.replay(64)
            elif agent == 'action_agent':
                action_agent.replay(64)
            elif agent == 'attacker_agent':
                attacker_agent.replay(64)

        print(f"Episode {episode+1} finished. Steps: {step_count}")
        print(f"Attacker steps taken: {len(episode_attacker)}")

        if (episode + 1) % 5 == 0:
            predictor_agent.update_target()
            action_agent.update_target()
            attacker_agent.update_target()
            print("Target networks updated.")

        rewards = env.rewards
        print(f"Episode {episode+1}: Rewards: {{'attacker_agent': {episode_reward_attacker}, 'predictor_agent': {episode_reward_predictor}, 'action_agent': {episode_reward_action}}}")

        predictor_history.append(episode_predictor)
        action_history.append(episode_action)
        attacker_history.append(episode_attacker)
        reward_history.append({
            'predictor_agent': episode_reward_predictor,
            'action_agent': episode_reward_action,
            'attacker_agent': episode_reward_attacker
        })

        # Chỉ cộng reward của predictor_agent và action_agent để lưu checkpoint
        total_reward = episode_reward_predictor + episode_reward_action
        if best_total_reward is None or total_reward > best_total_reward:
            best_total_reward = total_reward
            best_episode = episode + 1
            predictor_agent.save(predictor_ckpt_path)
            action_agent.save(action_ckpt_path)
            attacker_agent.save(attacker_ckpt_path)
            print(f"Saved best checkpoint at episode {best_episode} with total reward {best_total_reward}.")

        if (episode + 1) % 10 == 0:
            visualize_history(predictor_history, action_history, attacker_history, reward_history, episode)
            print(f"Best checkpoint so far: episode {best_episode} with total reward {best_total_reward}")

        elapsed = time.time() - start_time
        print(f"Episode {episode+1} time: {elapsed:.2f} seconds")
