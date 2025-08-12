import torch
import pandas as pd
from env.cloud_security_env import CloudSecurityEnv
from agents.base_dqn_agent import BaseDQNAgent

CHECKPOINT_DIR = "checkpoints"
predictor_ckpt_path = f"{CHECKPOINT_DIR}/predictor_agent_best.pth"
action_ckpt_path = f"{CHECKPOINT_DIR}/action_agent_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo môi trường với số bước nhỏ để test nhanh
env = CloudSecurityEnv(max_steps=100)

# Khởi tạo agent và load checkpoint
predictor_agent = BaseDQNAgent(
    action_space=env.action_spaces['predictor_agent'],
    obs_size=env.observation_spaces['predictor_agent'].shape[0],
    device=device
)
action_agent = BaseDQNAgent(
    action_space=env.action_spaces['action_agent'],
    obs_size=env.observation_spaces['action_agent'].shape[0],
    device=device
)

predictor_agent.load(predictor_ckpt_path)
action_agent.load(action_ckpt_path)
print("Loaded agents from checkpoint.")

env.reset()
results = []

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    done = termination or truncation

    if agent == 'predictor_agent':
        action = predictor_agent.act(observation)
    else:
        action = action_agent.act(observation)

    env.step(action)

    # Lưu kết quả dự đoán và reward
    if agent == 'action_agent':
        results.append({
            "step": env.idx,
            "predictor_action": env.last_prediction,
            "action_agent_action": action,
            "true_label": env.labels[env.idx-1] if env.idx > 0 else None,
            "reward": reward
        })

# Xuất kết quả ra file CSV
df = pd.DataFrame(results)
df.to_csv("inference_results.csv", index=False)
print("Inference finished. Results saved to inference_results.csv")