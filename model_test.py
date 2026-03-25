import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from env import UAVEnv, Config
from train import Agent  # 导入我们在 train.py 中定义的 Agent 类


class UAVEvaluator:
    def __init__(self, model_path):
        self.env = UAVEnv()
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.model = Agent(obs_dim, act_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def save_episode_gif(self, filename, max_steps=500):
        obs, _ = self.env.reset()
        frames = []
        positions = []

        print(f"🎬 正在生成录制: {filename}...")

        # --- 准备 3D 画布 ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for t in range(max_steps):
            # 推理动作 (使用均值，保证测试时动作稳定)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_mean = self.model.actor_mean(obs_tensor)
                action = action_mean.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = self.env.step(action)
            positions.append(self.env.pos.copy())

            if terminated or truncated:
                print(f"🏁 回合结束于步数: {t}, 奖励总计: {reward:.2f}")
                break

        # --- 动画渲染逻辑 ---
        pos_arr = np.array(positions)

        def update(num):
            ax.clear()
            # 1. 绘制障碍物
            r = self.cfg.OBS_RADIUS
            for p, obs_type in self.env.obstacles:
                if obs_type == 'sphere':
                    u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
                    x = p[0] + r * np.cos(u) * np.sin(v)
                    y = p[1] + r * np.sin(u) * np.sin(v)
                    z = p[2] + r * np.cos(v)
                    ax.plot_wireframe(x, y, z, color='red', alpha=0.1)
                else:
                    # 简化绘制立方体边缘
                    d = r
                    for i in [-d, d]:
                        for j in [-d, d]:
                            ax.plot([p[0] + i, p[0] + i], [p[1] + j, p[1] + j], [p[2] - d, p[2] + d], color='orange',
                                    alpha=0.2)
                            ax.plot([p[0] + i, p[0] + i], [p[1] - d, p[1] + d], [p[2] + j, p[2] + j], color='orange',
                                    alpha=0.2)
                            ax.plot([p[0] - d, p[0] + d], [p[1] + i, p[1] + i], [p[2] + j, p[2] + j], color='orange',
                                    alpha=0.2)

            # 2. 绘制起终点
            ax.scatter(self.env.start_pos[0], self.env.start_pos[1], 0, color='blue', s=100, label='Start')
            ax.scatter(self.env.goal_pos[0], self.env.goal_pos[1], 0, color='green', marker='*', s=200, label='Goal')

            # 3. 绘制飞行轨迹
            if num > 0:
                ax.plot(pos_arr[:num, 0], pos_arr[:num, 1], pos_arr[:num, 2], color='blue', lw=2)

            # 4. 绘制当前无人机位置 (质点)
            ax.scatter(pos_arr[num, 0], pos_arr[num, 1], pos_arr[num, 2], color='black', s=50)

            # 场景设置
            ax.set_xlim(0, 20);
            ax.set_ylim(0, 20);
            ax.set_zlim(0, 2.5)
            ax.set_title(f"UAV Path Planning - Step {num}")
            return ax,

        ani = FuncAnimation(fig, update, frames=len(pos_arr), interval=50, blit=False)
        writer = PillowWriter(fps=20)
        ani.save(filename, writer=writer)
        plt.close(fig)


# ================= 主程序 =================
if __name__ == "__main__":
    # 请确保这里填入你训练保存的最新模型文件名
    MODEL_FILE = "ppo_uav_1177600.pth"

    import os

    if not os.path.exists(MODEL_FILE):
        print(f"❌ 找不到模型文件 {MODEL_FILE}，请先运行 train.py 进行训练。")
    else:
        evaluator = UAVEvaluator(MODEL_FILE)
        for i in range(5):
            evaluator.save_episode_gif(f"uav_flight_{i + 1}.gif")
        print("✨ 5 个飞行演示 GIF 已全部生成完毕！")