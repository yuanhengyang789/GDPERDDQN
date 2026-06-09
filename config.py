import torch

# 超参数配置
BATCH_SIZE = 64
GAMMA = 0.9
MEMORY_SIZE = 10000
LEARNING_RATE = 0.005
NUM_EPISODES = 500
REPLAY_INTERVAL = 20

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
