<!-- markdownlint-disable MD060 -->
# GDPERDDQN

基于深度强化学习的2D栅格路径规划算法对比项目

## 环境要求

- Python 3.8+
- PyTorch 2.3.1 + CUDA 12.1
- 其他依赖见 `requirements.txt`

## 快速开始

配置好环境后运行 `main.py` 即可：

```bash
python main.py
```

## 项目结构

| 文件 | 说明 |
| ---- | ---- |
| `main.py` | 入口文件，组合三个算法并输出对比结果 |
| `config.py` | 超参数配置（BATCH_SIZE、GAMMA等）和设备配置 |
| `network.py` | DQN 网络结构定义 |
| `memory.py` | 经验回放类（普通回放、优先回放、双池回放） |
| `env.py` | 地图生成和三个 step 环境函数 |
| `utils.py` | 工具函数（动作选择、软更新、预训练、贝塞尔平滑、绘图） |
| `algorithm1.py` | 算法1 G-DPER-DDQN 训练流程 |
| `algorithm2.py` | 算法2 PER-DDQN 训练流程 |
| `algorithm3.py` | 算法3 ECMS-DDQN 训练流程 |

## 算法说明

代码会依次运行以下三个算法，并输出训练对比图和路径对比图：

| 算法 | 核心特点 |
| ---- | -------- |
| **G-DPER-DDQN** | 双池经验回放+动态采样比+网络预训练+Huber损失 |
| **PER-DDQN** | 优先经验回放（基线算法） |
| **ECMS-DDQN** | 障碍物感知双池回放+N步引导(3步)+转向/振荡惩罚 |

## 路径可视化

- **G-DPER-DDQN**：路径经过贝塞尔曲线平滑处理，绘制时向左上方偏移
- **PER-DDQN**：使用原始离散路径，绘制时向右侧偏移
- **ECMS-DDQN**：使用原始离散路径，绘制时向右下方偏移

偏移用于避免三条路径重叠，便于直观对比。

## 输出文件

运行结束后会生成：

- 三个训练模型文件（`.pth`）
- 三个训练数据CSV文件（`training_rewards.csv`、`training_steps.csv`、`training_times.csv`）
- 训练曲线对比图和路径对比图

## 自定义配置

超参数集中在 `config.py` 中，可直接修改。开发人员还可自由修改地图大小和障碍物比例，修改地图大小时需注意：

1. 修改 `network.py` 中 `DQN` 的输出层维度以适应新地图
2. 修改 `env.py` 中 `step` 函数的边界检查代码
3. 调整 `config.py` 中每轮最大步数限制

## 预训练模型

三个 `.pth` 文件为作者在 20×20 地图、障碍物比例 0.2 下训练的模型。

## 后续计划

作者目前正在研究在 ROS2 + Gazebo 环境下搭建仿真模型，相关代码即将发布。
