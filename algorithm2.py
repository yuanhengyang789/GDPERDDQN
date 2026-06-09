import torch.optim as optim
import time
from config import device, BATCH_SIZE, MEMORY_SIZE, LEARNING_RATE, NUM_EPISODES, REPLAY_INTERVAL
from network import DQN
from memory import PrioritizedReplayMemoryV1
from utils import matrix_to_img, choose_action, soft_update, optimize_model_v2, test_net
from env import step_v2


def run_algorithm_v2(map, start_pos, target_pos):
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = PrioritizedReplayMemoryV1(MEMORY_SIZE, alpha=0.6, beta_start=0.4)
    epsilon = 0.5
    eps_decay = 0.99
    min_epsilon = 0.05

    steps_done = 0
    episode_steps = []
    total_rewards = []
    cumulative_times = []
    cumulative_time = 0

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
            next_pos, reward, done, visited_positions = step_v2(
                current_pos, action, target_pos, map, visited_positions, prev_action)
            next_state = matrix_to_img(next_pos, map).to(device) if not done else None
            memory.push(state, action, reward, next_state, done)
            prev_action = action

            if steps_done % REPLAY_INTERVAL == 0:
                if len(memory) >= BATCH_SIZE:
                    optimize_model_v2(policy_net, target_net, optimizer, memory, beta=0.4)
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
        episode_time = time.time() - episode_start_time
        cumulative_time += episode_time
        cumulative_times.append(cumulative_time)

        if episode % 1 == 0:
            print(f'Algorithm 2 (PER-DDQN) - Episode {episode}, Steps: {step_count}, '
                  f'Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}, '
                  f'Memory: {len(memory)}')

    final_path = test_net(policy_net, map, start_pos, target_pos, step_v2)
    return episode_steps, total_rewards, cumulative_times, final_path, policy_net
