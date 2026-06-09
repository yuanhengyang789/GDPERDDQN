import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from config import device, BATCH_SIZE, GAMMA, MEMORY_SIZE, LEARNING_RATE, NUM_EPISODES, REPLAY_INTERVAL
from network import DQN
from memory import DualReplayMemory
from utils import matrix_to_img, choose_action, soft_update, initialize_network_weights, test_net
from env import step_v1


def run_algorithm_v1(map, start_pos, target_pos):
    policy_net = DQN().to(device)
    target_net = DQN().to(device)

    # 使用预训练值初始化网络
    pretrain_start = time.time()
    initialize_network_weights(policy_net, map, target_pos)
    pretrain_duration = time.time() - pretrain_start
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    epsilon = 0.5
    eps_decay = 0.99
    min_epsilon = 0.05
    normal_capacity = int(MEMORY_SIZE * 0.6)
    elite_capacity = MEMORY_SIZE - normal_capacity
    memory = DualReplayMemory(normal_capacity, elite_capacity)

    steps_done = 0
    episode_steps = []
    total_rewards = []
    cumulative_times = []
    cumulative_time = 0
    epsilons = []
    training_start = time.time()

    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        current_pos = start_pos
        total_reward = 0
        step_count = 0
        visited_positions = {}
        prev_action = None

        while True:
            state = matrix_to_img(current_pos, map).to(device)
            action = choose_action(state, policy_net, epsilon)
            next_pos, reward, done, visited_positions = step_v1(
                current_pos, action, target_pos, map, visited_positions, prev_action)
            next_state = matrix_to_img(next_pos, map).to(device) if not done else None
            memory.push(state, action, reward, next_state, done)
            prev_action = action

            if steps_done % REPLAY_INTERVAL == 0 and len(memory) >= BATCH_SIZE:
                batch, _ = memory.sample(BATCH_SIZE)
                if batch:
                    state_batch = torch.cat([item[0].to(device) for item in batch])
                    action_batch = torch.tensor([item[1] for item in batch], device=device).unsqueeze(1)
                    reward_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
                    non_final_mask = torch.tensor([item[3] is not None for item in batch], device=device, dtype=torch.bool)
                    non_final_next_states = torch.cat([item[3] for item in batch if item[3] is not None]).to(device)
                    current_q_values = policy_net(state_batch).gather(1, action_batch)
                    next_q_values = torch.zeros(len(batch), device=device)
                    with torch.no_grad():
                        if len(non_final_next_states) > 0:
                            next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                            next_q_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()
                    target_q_values = reward_batch + (GAMMA * next_q_values)
                    loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
                    # 记录各池损失用于动态采样比例调整
                    td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
                    normal_size = memory._last_normal_size
                    memory.record_losses(
                        td_errors[:normal_size].tolist() if normal_size > 0 else [],
                        td_errors[normal_size:].tolist() if normal_size < len(batch) else []
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
                    optimizer.step()
                    soft_update(target_net, policy_net, tau=0.01)

            current_pos = next_pos
            step_count += 1
            steps_done += 1
            total_reward += reward

            if done or step_count >= 3000:
                episode_steps.append(step_count)
                total_rewards.append(total_reward)
                break

        if episode < 50:
            epsilon = epsilon
        else:
            epsilon = max(min_epsilon, epsilon * eps_decay)
        epsilons.append(epsilon)
        episode_time = time.time() - episode_start_time
        cumulative_time += episode_time
        cumulative_times.append(cumulative_time)

        if episode % 1 == 0:
            stats = memory.get_memory_stats()
            print(f'Algorithm 1 - Episode {episode}, Steps: {step_count}, '
                  f'Reward: {total_reward:.1f}, '
                  f'Elite/Normal: {stats["elite_size"]}/{stats["normal_size"]}, '
                  f'Sampling Ratio: {stats["normal_ratio"]:.2f}/{1 - stats["normal_ratio"]:.2f}, '
                  f'Epsilon: {epsilon:.3f}')

    training_duration = time.time() - training_start
    final_path = test_net(policy_net, map, start_pos, target_pos, step_v1)
    print(f"Algorithm 1 Pretraining Time: {pretrain_duration:.6f} 秒")

    return episode_steps, total_rewards, cumulative_times, final_path, policy_net, epsilons, training_duration
