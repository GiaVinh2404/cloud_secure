def train_episode(env, predictor_agent, action_agent, batch_size=64, target_update_freq=5):
    env.reset()

    last_obs = {}
    last_action = {}

    step_count = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if agent == 'predictor_agent':
            action = predictor_agent.act(observation)
        else:
            action = action_agent.act(observation)

        # Nếu đã có state trước đó, lưu kinh nghiệm
        if agent in last_obs:
            if agent == 'predictor_agent':
                predictor_agent.remember(last_obs[agent], last_action[agent], reward, observation, done)
                # Replay với batch lớn hơn và sau mỗi bước
                predictor_agent.replay(batch_size)
            else:
                action_agent.remember(last_obs[agent], last_action[agent], reward, observation, done)
                action_agent.replay(batch_size)

        last_obs[agent] = observation
        last_action[agent] = action

        env.step(action)
        step_count += 1

        # Cập nhật target network thường xuyên hơn
        if step_count % target_update_freq == 0:
            predictor_agent.update_target()
            action_agent.update_target()

    # Trả về reward tổng hợp cho cả hai agent
    return env.rewards