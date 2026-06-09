import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pandas as pd

from env import generate_map
from utils import smooth_path, plot_paths_four
from algorithm1 import run_algorithm_v1
from algorithm2 import run_algorithm_v2
from algorithm3 import run_algorithm_v3


def main():
    # 生成20×20地图
    map_array = generate_map(size=20, obstacle_ratio=0.2)
    start_pos = (19, 0)
    target_pos = (0, 19)

    # 运行三个算法获取训练好的网络
    print("Running Algorithm 1 (G-DPER-DDQN)...")
    steps1, rewards1, times1, path1, net1, epsilons1, training_duration1 = run_algorithm_v1(map_array, start_pos, target_pos)
    print(f"算法1运行时间: {training_duration1:.6f} 秒")
    torch.save(net1.state_dict(), "g_dper_ddqn_model.pth")
    print("G-DPER-DDQN 模型已保存为 g_dper_ddqn_model.pth")

    print("\nRunning Algorithm 2 (PER-DDQN)...")
    start2 = time.time()
    steps2, rewards2, times2, path2, net2 = run_algorithm_v2(map_array, start_pos, target_pos)
    end2 = time.time()
    print(f"算法2运行时间: {end2 - start2:.6f} 秒")
    torch.save(net2.state_dict(), "per_ddqn_model.pth")
    print("PER-DDQN 模型已保存为 per_ddqn_model.pth")

    print("\nRunning Algorithm 3 (ECMS-DDQN)...")
    start3 = time.time()
    steps3, rewards3, times3, path3, net3 = run_algorithm_v3(map_array, start_pos, target_pos)
    end3 = time.time()
    print(f"算法3运行时间: {end3 - start3:.6f} 秒")
    torch.save(net3.state_dict(), "ECMSddqn_model.pth")
    print("ECMS-DDQN 模型已保存为 ECMSddqn_model.pth")

    # 处理奖励值
    rewards1 = np.clip(np.array(rewards1), -2000, None)
    rewards2 = np.clip(np.array(rewards2), -2000, None)
    rewards3 = np.clip(np.array(rewards3), -2000, None)

    # 绘制图表
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(steps1, label='Alg1: G-DPER-DDQN')
    plt.plot(steps2, label='Alg2: PER-DDQN')
    plt.plot(steps3, label='Alg3: ECMS-DDQN')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(rewards1, label='Alg1: G-DPER-DDQN')
    plt.plot(rewards2, label='Alg2: PER-DDQN')
    plt.plot(rewards3, label='Alg3: ECMS-DDQN')
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(times1, label='Alg1: G-DPER-DDQN')
    plt.plot(times2, label='Alg2: PER-DDQN')
    plt.plot(times3, label='Alg3: ECMS-DDQN')
    plt.title('Cumulative Time')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 绘制epsilon变化图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epsilons1) + 1), epsilons1, 'b-', label='Epsilon')
    plt.title('Epsilon Changes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.show()

    # 路径对比（算法1贝塞尔平滑，算法2和3原始路径+偏移）
    smooth_path1 = smooth_path(path1)
    plot_paths_four(smooth_path1, path2, path3,
                    "Paths Comparison",
                    start_pos, target_pos, map_array)

    print(f"G-DPER-DDQN 路径长度: {len(path1)}")
    print(f"PER-DDQN 路径长度: {len(path2)}")
    print(f"ECMS-DDQN 路径长度: {len(path3)}")

    # 保存训练数据到CSV文件
    steps_data = {
        'Episode': range(len(steps1)),
        'G-DPER-DDQN': steps1,
        'PER-DDQN': steps2,
        'ECMS-DDQN': steps3
    }
    rewards_data = {
        'Episode': range(len(rewards1)),
        'G-DPER-DDQN': rewards1,
        'PER-DDQN': rewards2,
        'ECMS-DDQN': rewards3
    }
    times_data = {
        'Episode': range(len(times1)),
        'G-DPER-DDQN': times1,
        'PER-DDQN': times2,
        'ECMS-DDQN': times3
    }

    pd.DataFrame(steps_data).to_csv('training_steps.csv', index=False)
    pd.DataFrame(rewards_data).to_csv('training_rewards.csv', index=False)
    pd.DataFrame(times_data).to_csv('training_times.csv', index=False)

    print("\nTraining data has been saved to:")
    print("- training_steps.csv")
    print("- training_rewards.csv")
    print("- training_times.csv")


if __name__ == "__main__":
    main()
