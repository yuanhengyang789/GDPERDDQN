import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from collections import deque
from config import device, BATCH_SIZE, GAMMA, MEMORY_SIZE, LEARNING_RATE, NUM_EPISODES, REPLAY_INTERVAL
from network import DQN
from memory import DualReplayMemoryObstacle
from utils import matrix_to_img, choose_action, soft_update, test_net
from env import step_v3


def run_algorithm_v3(map, start_pos, target_pos):
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = DualReplayMemoryObstacle(near_capacity=int(MEMORY_SIZE * 0.3), all_capacity=int(MEMORY_SIZE * 0.7))

    steps_done = 0
    episode_steps = []
    total_rewards = []
    cumulative_times = []
    cumulative_time = 0
    epsilon = 0.5
    N_STEPS = 3
    eps_decay = 0.99
    min_epsilon = 0.05

    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        current_pos = start_pos
        total_reward = 0
        step_count = 0
        visited_positions = {}
        prev_action = None
        prev_actions = []
        n_step_buffer = deque(maxlen=N_STEPS)

        while True:
            state = matrix_to_img(current_pos, map).to(device)
            action = choose_action(state, policy_net, epsilon)
            next_pos, reward, done, visited_positions, prev_actions = step_v3(
                current_pos, action, target_pos, map, visited_positions, prev_action, prev_actions)
            next_state = matrix_to_img(next_pos, map).to(device) if not done else None
            n_step_buffer.append((state, action, reward, next_state, done, current_pos))

            if len(n_step_buffer) == N_STEPS:
                n_reward, n_next_state, n_done = 0, None, False
                for idx, (_, _, r, ns, d, _) in enumerate(n_step_buffer):
                    n_reward += (GAMMA ** idx) * r
                    if d:
                        n_done = True
                        n_next_state = ns
                        break
                    else:
                        n_next_state = ns
                first_state, first_action, _, _, _, first_pos = n_step_buffer[0]
                memory.push(first_state, first_action, n_reward, n_next_state, n_done, first_pos, map)

            prev_action = action
            if steps_done % REPLAY_INTERVAL == 0:
                batch, _, _ = memory.sample(BATCH_SIZE)
                if batch:
                    batch = list(batch)
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
                    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
                    optimizer.step()
                    priorities = (torch.abs(current_q_values.squeeze() - target_q_values) + 1e-5).detach().cpu().numpy()
                    memory.update_priorities(None, priorities)
                    soft_update(target_net, policy_net, tau=0.01)

            current_pos = next_pos
            step_count += 1
            steps_done += 1
            total_reward += reward

            if done or step_count >= 3000:
                while len(n_step_buffer) > 0:
                    n_reward, n_next_state, n_done = 0, None, False
                    for idx, (_, _, r, ns, d, _) in enumerate(n_step_buffer):
                        n_reward += (GAMMA ** idx) * r
                        if d:
                            n_done = True
                            n_next_state = ns
                            break
                        else:
                            n_next_state = ns
                    first_state, first_action, _, _, _, first_pos = n_step_buffer[0]
                    is_last = len(n_step_buffer) == 1
                    memory.push(first_state, first_action, n_reward, n_next_state, n_done, first_pos, map, is_episode_end=is_last)
                    n_step_buffer.popleft()
                episode_steps.append(step_count)
                total_rewards.append(total_reward)
                break

        if episode < 50:
            epsilon = epsilon
        else:
            epsilon = max(min_epsilon, epsilon * eps_decay)
        episode_time = time.time() - episode_start_time
        cumulative_time += episode_time
        cumulative_times.append(cumulative_time)

        if episode % 1 == 0:
            print(f'Algorithm 3 - Episode {episode}, Steps: {step_count}, '
                  f'Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}, '
                  f'Memory: {len(memory)}, Near Ratio: {memory.near_ratio:.2f}')

    final_path = test_net(policy_net, map, start_pos, target_pos, step_v3)
    return episode_steps, total_rewards, cumulative_times, final_path, policy_net
