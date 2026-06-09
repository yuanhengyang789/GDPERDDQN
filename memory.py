import numpy as np
import random
from config import NUM_EPISODES


class ReplayMemory:
    """普通经验回放缓冲区"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch, None, np.ones(batch_size)

    def __len__(self):
        return len(self.memory)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
        self.max_priority = 1.0

    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(self.max_priority, priority)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, change)

    def get_leaf(self, value):
        idx = self.retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total_priority(self):
        return max(self.tree[0], 1e-8)

    def get_min_priority(self):
        if self.size == 0:
            return 1.0
        non_zero_priorities = self.tree[self.capacity-1:self.capacity-1+self.size]
        non_zero_priorities = non_zero_priorities[non_zero_priorities > 0]
        if len(non_zero_priorities) == 0:
            return 1.0
        return np.min(non_zero_priorities)


class PrioritizedReplayMemoryV1:
    """优先经验回放（算法2 PER-DDQN专用）"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 1
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.tree.max_priority, 1.0)
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = min(1.0, self.beta + self.frame_idx * (1.0 - self.beta) / self.beta_frames)
            self.frame_idx += 1

        if self.tree.size == 0:
            return [], [], np.array([])

        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        total_priority = self.tree.total_priority()
        min_priority = self.tree.get_min_priority()
        min_prob = min_priority / total_priority if total_priority > 0 else 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            if a == b:
                b = a + 1e-8
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            prob = priority / total_priority if total_priority > 0 else 1.0
            if min_prob > 0:
                weight = (prob / min_prob) ** (-beta)
            else:
                weight = 1.0
            batch.append(data)
            indices.append(index)
            priorities.append(weight)

        weights = np.array(priorities, dtype=np.float32)
        weights = weights / weights.max() if weights.max() > 0 else weights
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = np.power(priorities + self.epsilon, self.alpha)
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


class DualReplayMemoryObstacle:
    """障碍物感知双经验池（算法3 ECMS-DDQN专用）"""
    def __init__(self, near_capacity, all_capacity, p0=0.3, p1=0.6, beta_t=0.4, total_episodes=NUM_EPISODES):
        self.near_memory = ReplayMemory(near_capacity)
        self.all_memory = ReplayMemory(all_capacity)
        self.near_ratio = 0.4
        self.beta_t = 0.4
        self.min_ratio = 0
        self.max_ratio = 0.6
        self.p0 = p0
        self.p1 = p1
        self.beta_t = beta_t
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.epsilon_t = 1.0
        self.near_losses = []
        self.all_losses = []

    def is_near_obstacle(self, pos, map_array):
        r, c = pos
        size = map_array.shape[0]
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if map_array[nr, nc] == 1:
                        return True
        return False

    def push(self, state, action, reward, next_state, done, pos, map_array, is_episode_end=False):
        self.all_memory.push(state, action, reward, next_state, done)
        if self.is_near_obstacle(pos, map_array):
            self.near_memory.push(state, action, reward, next_state, done)
        if done and is_episode_end:
            self.current_episode += 1
            self.adjust_sampling_ratio()

    def adjust_sampling_ratio(self):
        t = self.current_episode / self.total_episodes
        self.epsilon_t = max(0.01, self.epsilon_t * 0.995)
        l0 = np.mean(self.all_losses) if self.all_losses else 0
        l1 = np.mean(self.near_losses) if self.near_losses else 0
        total_loss = l0 + l1 if (l0 + l1) > 0 else 1
        if t < self.beta_t:
            self.near_ratio = (self.p0 * self.epsilon_t + self.p1 * (l1 / total_loss))
        else:
            self.near_ratio = 0
        self.near_ratio = max(self.min_ratio, min(self.max_ratio, self.near_ratio))
        self.all_losses = []
        self.near_losses = []

    def sample(self, batch_size, beta=0.4):
        near_size = int(batch_size * self.near_ratio)
        all_size = batch_size - near_size
        near_size = min(near_size, len(self.near_memory))
        all_size = min(all_size, len(self.all_memory))
        if near_size == 0 and all_size == 0:
            self.last_indices_type = "empty"
            return [], None, np.array([])
        if near_size == 0:
            self.last_indices_type = "all_only"
            batch, _, _ = self.all_memory.sample(all_size)
            return batch, None, np.ones(len(batch))
        if all_size == 0:
            self.last_indices_type = "near_only"
            batch, _, _ = self.near_memory.sample(near_size)
            return batch, None, np.ones(len(batch))
        near_batch, _, _ = self.near_memory.sample(near_size)
        all_batch, _, _ = self.all_memory.sample(all_size)
        batch = near_batch + all_batch
        self.last_indices_type = "mixed"
        self.last_near_size = near_size
        return batch, None, np.ones(len(batch))

    def update_priorities(self, indices, priorities):
        if self.last_indices_type == "near_only":
            self.near_losses.extend(priorities)
        elif self.last_indices_type == "all_only":
            self.all_losses.extend(priorities)
        elif self.last_indices_type == "mixed":
            self.near_losses.extend(priorities[:self.last_near_size])
            self.all_losses.extend(priorities[self.last_near_size:])

    def __len__(self):
        return len(self.near_memory) + len(self.all_memory)


class DualReplayMemory:
    """双经验池（普通均匀采样，算法1 G-DPER-DDQN专用）"""
    def __init__(self, normal_capacity, elite_capacity, elite_threshold=2, p0=0.4, p1=0.5, beta_t=0.4):
        self.normal_memory = ReplayMemory(normal_capacity)
        self.elite_memory = ReplayMemory(elite_capacity)
        self.elite_threshold = elite_threshold
        self.normal_ratio = 0.5
        self.last_indices_type = None
        self.min_ratio = 0.3
        self.max_ratio = 0.8
        self.p0 = p0
        self.p1 = p1
        self.beta_t = beta_t
        self.total_episodes = NUM_EPISODES
        self.current_episode = 0
        self.epsilon_t = 1.0
        self.normal_losses = []
        self.elite_losses = []

    def push(self, state, action, reward, next_state, done):
        if reward >= self.elite_threshold:
            self.elite_memory.push(state, action, reward, next_state, done)
        else:
            self.normal_memory.push(state, action, reward, next_state, done)
        if done:
            self.current_episode += 1
            if self.current_episode % 10 == 0:
                self.adjust_sampling_ratio()

    def adjust_sampling_ratio(self):
        t = self.current_episode / self.total_episodes
        self.epsilon_t = max(0.01, self.epsilon_t * 0.995)
        l0 = np.mean(self.normal_losses) if self.normal_losses else 0
        l1 = np.mean(self.elite_losses) if self.elite_losses else 0
        total_loss = l0 + l1 if (l0 + l1) > 0 else 1
        if t < self.beta_t:
            self.normal_ratio = (self.p0 * self.epsilon_t + self.p1 * (l0 / total_loss))
        else:
            self.normal_ratio = 0.4
        self.normal_ratio = max(self.min_ratio, min(self.max_ratio, self.normal_ratio))
        self.normal_losses = []
        self.elite_losses = []

    def record_losses(self, normal_losses, elite_losses):
        if normal_losses:
            self.normal_losses.extend(normal_losses)
        if elite_losses:
            self.elite_losses.extend(elite_losses)

    def get_memory_stats(self):
        return {
            'normal_size': len(self.normal_memory),
            'elite_size': len(self.elite_memory),
            'normal_ratio': self.normal_ratio
        }

    def sample(self, batch_size):
        if len(self.normal_memory) == 0 and len(self.elite_memory) == 0:
            self.last_indices_type = "empty"
            return [], np.array([])

        normal_size = int(batch_size * self.normal_ratio)
        elite_size = batch_size - normal_size

        if (len(self.normal_memory) < normal_size) and (len(self.elite_memory) < elite_size):
            normal_size = len(self.normal_memory)
            elite_size = len(self.elite_memory)
        if len(self.normal_memory) < normal_size:
            normal_size = len(self.normal_memory)
            elite_size = min(batch_size - normal_size, len(self.elite_memory))
        if len(self.elite_memory) < elite_size:
            elite_size = len(self.elite_memory)
            normal_size = min(batch_size - elite_size, len(self.normal_memory))

        if normal_size == 0:
            self.last_indices_type = "elite_only"
            batch, _, weights = self.elite_memory.sample(elite_size)
            self._last_normal_size = 0
            return batch, weights
        if elite_size == 0:
            self.last_indices_type = "normal_only"
            batch, _, weights = self.normal_memory.sample(normal_size)
            self._last_normal_size = normal_size
            return batch, weights

        normal_batch, _, normal_weights = self.normal_memory.sample(normal_size)
        elite_batch, _, elite_weights = self.elite_memory.sample(elite_size)
        batch = normal_batch + elite_batch
        self.last_indices_type = "mixed"
        self._last_normal_size = normal_size
        weights = np.concatenate((normal_weights, elite_weights))
        return batch, weights

    def __len__(self):
        return len(self.normal_memory) + len(self.elite_memory)
