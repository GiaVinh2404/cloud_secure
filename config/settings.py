# config/settings.py

DQN_CONFIG = {
    "lr": 5e-5,             # learning rate giảm xuống để ổn định hơn
    "gamma": 0.99,          # discount factor giữ nguyên
    "epsilon_start": 1.0,   # initial epsilon
    "epsilon_min": 0.05,    # tăng min epsilon để giữ mức explore
    "epsilon_decay": 0.98, # decay chậm hơn -> explore lâu hơn
    "batch_size": 128,      # giảm batch size để update đa dạng hơn
    "memory_size": 50000,   # replay buffer size giữ nguyên
    "target_update": 500,   # update target thường xuyên hơn
    "hidden_layers": [256, 128, 64],  # đơn giản hoá DNN, tránh overfitting
    "max_steps": 500,       # maximum steps per episode giữ nguyên
    "num_episodes": 500,
}
