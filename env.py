import numpy as np
from queue import Queue


def generate_map(size=20, obstacle_ratio=0.2):
    def is_path_exists(map_array, start, end):
        """使用BFS检查是否存在可达路径"""
        queue = Queue()
        visited = set()
        queue.put(start)
        visited.add(start)
        while not queue.empty():
            current = queue.get()
            if current == end:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if (0 <= nr < size and 0 <= nc < size and
                        (nr, nc) not in visited and map_array[nr, nc] == 0):
                    queue.put((nr, nc))
                    visited.add((nr, nc))
        return False

    while True:
        map_array = np.zeros((size, size), dtype=np.float32)
        start_pos = (size - 1, 0)  # 左下角
        target_pos = (0, size - 1)  # 右上角
        num_obstacles = int(size * size * obstacle_ratio)
        possible_positions = [(r, c) for r in range(size) for c in range(size)
                              if (r, c) != start_pos and (r, c) != target_pos]
        obstacle_positions = np.random.choice(len(possible_positions), size=num_obstacles, replace=False)
        for idx in obstacle_positions:
            r, c = possible_positions[idx]
            map_array[r, c] = 1
        if is_path_exists(map_array, start_pos, target_pos):
            return map_array


def step_v1(current_pos, action, target_pos, map, visited_positions=None, prev_action=None):
    """算法1 G-DPER-DDQN 的环境步进函数"""
    if visited_positions is None:
        visited_positions = {}
    row, col = current_pos
    actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    dr, dc = actions[action]
    n_row = max(0, min(19, row + dr))
    n_col = max(0, min(19, col + dc))
    done = False
    base_reward = -1
    new_pos = (n_row, n_col)

    if new_pos not in visited_positions:
        reward = 5
        visited_positions[new_pos] = 1
    else:
        prev_euclidean_distance = np.sqrt((row - target_pos[0]) ** 2 + (col - target_pos[1]) ** 2)
        current_euclidean_distance = np.sqrt((n_row - target_pos[0]) ** 2 + (n_col - target_pos[1]) ** 2)
        distance_reward = 10 * (prev_euclidean_distance - current_euclidean_distance)
        repeat_penalty = -3
        reward = distance_reward + repeat_penalty + base_reward

    if (n_row, n_col) == (row, col) or map[n_row, n_col] == 1:
        reward = -5
        return (row, col), reward, done, visited_positions
    if (n_row, n_col) == target_pos:
        reward = 50
        done = True
        return (n_row, n_col), reward, done, visited_positions
    return (n_row, n_col), reward, done, visited_positions


def step_v2(current_pos, action, target_pos, map, visited_positions=None, prev_action=None):
    """算法2 PER-DDQN 的环境步进函数"""
    if visited_positions is None:
        visited_positions = {}
    row, col = current_pos
    actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    dr, dc = actions[action]
    n_row = max(0, min(19, row + dr))
    n_col = max(0, min(19, col + dc))
    done = False
    base_reward = -1
    new_pos = (n_row, n_col)

    if new_pos not in visited_positions:
        reward = 0
        visited_positions[new_pos] = 1
    else:
        reward = base_reward

    if (n_row, n_col) == (row, col) or map[n_row, n_col] == 1:
        reward = -5
        return (row, col), reward, done, visited_positions
    if (n_row, n_col) == target_pos:
        reward = 20
        done = True
        return (n_row, n_col), reward, done, visited_positions
    return (n_row, n_col), reward, done, visited_positions


def step_v3(current_pos, action, target_pos, map, visited_positions=None, prev_action=None, prev_actions=None):
    """算法3 ECMS-DDQN 的环境步进函数"""
    if visited_positions is None:
        visited_positions = {}
    if prev_actions is None:
        prev_actions = []
    row, col = current_pos
    actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    dr, dc = actions[action]
    n_row = max(0, min(19, row + dr))
    n_col = max(0, min(19, col + dc))
    done = False
    new_pos = (n_row, n_col)
    size = map.shape[0]

    reward = 0
    if new_pos not in visited_positions:
        visited_positions[new_pos] = 1
    else:
        prev_euclidean_distance = np.sqrt((row - target_pos[0]) ** 2 + (col - target_pos[1]) ** 2)
        current_euclidean_distance = np.sqrt((n_row - target_pos[0]) ** 2 + (n_col - target_pos[1]) ** 2)
        distance_reward = 5 * (prev_euclidean_distance - current_euclidean_distance)
        reward += distance_reward

    # 转弯惩罚
    if prev_action is not None and action != prev_action:
        reward -= 0.1

    # 靠近障碍物惩罚
    near_obstacle = False
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = n_row + dr, n_col + dc
            if 0 <= nr < size and 0 <= nc < size:
                if map[nr, nc] == 1:
                    near_obstacle = True
                    break
        if near_obstacle:
            break
    if near_obstacle:
        reward -= 0.5

    # 震荡惩罚
    prev_actions = (prev_actions + [action])[-10:]
    turn_count = 0
    for i in range(1, len(prev_actions)):
        if prev_actions[i] != prev_actions[i - 1]:
            turn_count += 1
    if turn_count >= 3:
        reward -= 0.5 * (turn_count - 2)

    # 撞墙或障碍惩罚
    if (n_row, n_col) == (row, col) or map[n_row, n_col] == 1:
        reward = -5
        return (row, col), reward, done, visited_positions, prev_actions

    # 到达目标奖励
    if (n_row, n_col) == target_pos:
        reward = 20
        done = True
        return (n_row, n_col), reward, done, visited_positions, prev_actions

    return (n_row, n_col), reward, done, visited_positions, prev_actions
