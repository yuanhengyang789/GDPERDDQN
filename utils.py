import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from scipy.special import comb
from config import device, BATCH_SIZE, GAMMA, LEARNING_RATE


def matrix_to_img(pos, map_array):
    """将位置转换为网络输入张量"""
    row, col = pos
    state = map_array.copy()
    state[row, col] = 2
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


def choose_action(state, policy_net, epsilon):
    """ε-贪婪策略选择动作"""
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1).item()


def soft_update(target_net, policy_net, tau):
    """软更新目标网络"""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


def test_net(policy_net, map, start_pos, target_pos, step_func):
    """使用训练好的网络测试路径"""
    current_pos = start_pos
    path = [current_pos]
    prev_action = None
    prev_actions = []
    visited_positions = {}
    for _ in range(100):
        state = matrix_to_img(current_pos, map).to(device)
        with torch.no_grad():
            action = policy_net(state).max(1)[1].item()

        if step_func.__name__ == "step_v3":
            result = step_func(current_pos, action, target_pos, map, visited_positions, prev_action, prev_actions)
            if isinstance(result, tuple) and len(result) == 5:
                next_pos, reward, done, visited_positions, prev_actions = result
            else:
                next_pos, reward, done = result
        else:
            result = step_func(current_pos, action, target_pos, map, visited_positions, prev_action)
            if isinstance(result, tuple) and len(result) == 4:
                next_pos, reward, done, visited_positions = result
            else:
                next_pos, reward, done = result

        path.append(next_pos)
        if done:
            break
        prev_action = action
        current_pos = next_pos
    return path


def initialize_q_values(map_array, target_pos):
    """计算启发式Q值"""
    rows, cols = map_array.shape
    q_values = np.zeros((rows, cols, 4))
    dr_dc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(rows):
        for c in range(cols):
            for action in range(4):
                nr, nc = r + dr_dc[action][0], c + dr_dc[action][1]
                if 0 <= nr < rows and 0 <= nc < cols:
                    dist = np.sqrt((nr - target_pos[0]) ** 2 + (nc - target_pos[1]) ** 2)
                    q_values[r, c, action] = np.exp(-dist)
                else:
                    q_values[r, c, action] = 0
    return q_values


def initialize_network_weights(net, map_array, target_pos, epochs=1, lr=0.0001):
    """使用监督学习预训练网络"""
    q_values = initialize_q_values(map_array, target_pos)
    rows, cols = map_array.shape
    samples = []
    for r in range(rows):
        for c in range(cols):
            state = matrix_to_img((r, c), map_array)
            target_q = torch.tensor(q_values[r, c], dtype=torch.float32, device=device)
            samples.append((state, target_q))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0
        for state, target_q in samples:
            pred_q = net(state)
            loss = F.mse_loss(pred_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 1 == 0:
            avg_loss = total_loss / len(samples)
            print(f'Pretraining Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.6f}')
    print(f'Pretraining completed with final avg loss: {total_loss / len(samples):.6f}')


def optimize_model_v2(policy_net, target_net, optimizer, memory, beta=0.4):
    """算法2 PER-DDQN 的优化函数"""
    if len(memory) < BATCH_SIZE:
        return

    transitions, indices, is_weights = memory.sample(BATCH_SIZE, beta)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.tensor(batch[1], device=device, dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)
    non_final_mask = torch.tensor([s is not None for s in batch[3]], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch[3] if s is not None])

    next_q_values = torch.zeros(BATCH_SIZE, device=device)
    if len(non_final_next_states) > 0:
        with torch.no_grad():
            next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_q_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()

    target_q_values = reward_batch + (GAMMA * next_q_values)
    current_q_values = policy_net(state_batch).gather(1, action_batch)

    td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors + 1e-5)

    is_weights = torch.tensor(is_weights, device=device, dtype=torch.float32)
    loss = (is_weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def smooth_path(path, num_points=100):
    """使用贝塞尔曲线对路径进行平滑处理"""
    if len(path) < 4:
        return path

    x = [p[1] for p in path]
    y = [p[0] for p in path]
    n = len(x) - 1

    try:
        t_new = np.linspace(0, 1, num_points)
        t_col = t_new.reshape(-1, 1)
        i_row = np.arange(n + 1).reshape(1, -1)
        bernstein = comb(n, i_row) * (t_col ** i_row) * ((1 - t_col) ** (n - i_row))
        x_smooth = bernstein @ np.array(x)
        y_smooth = bernstein @ np.array(y)
        return list(zip(y_smooth, x_smooth))
    except:
        return path


def plot_paths_four(path1, path2, path3, title, start_pos, target_pos, map):
    """绘制三条路径对比图"""
    plt.figure(figsize=(10, 10))
    plt.imshow(map, cmap='gray_r', origin='lower')

    offset = 0.15
    plt.plot([p[1] - offset for p in path1], [p[0] - offset for p in path1], 'r-',
             label='G-DPER-DDQN', linewidth=1.5)
    plt.plot([p[1] + offset for p in path2], [p[0] for p in path2], 'g-',
             label='PER-DDQN', linewidth=1.5)
    plt.plot([p[1] + offset for p in path3], [p[0] - offset for p in path3], 'b-',
             label='ECMS-DDQN', linewidth=1.5)

    plt.scatter(start_pos[1], start_pos[0], c='blue', s=200, label='Start')
    plt.scatter(target_pos[1], target_pos[0], c='red', s=200, label='Target')

    ax = plt.gca()
    ax.set_xticks(np.arange(0, map.shape[1], 1) - 0.5)
    ax.set_yticks(np.arange(0, map.shape[0], 1) - 0.5)
    ax.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    plt.title(title)
    plt.legend()
    plt.show()
