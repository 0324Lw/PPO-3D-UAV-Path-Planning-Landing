import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import UAVEnv, Config


class EnvTester:
    def __init__(self):
        self.env = UAVEnv()
        self.cfg = Config()

    def test_spaces_and_step(self):
        """1. 测试空间维度与数值输出"""
        print("=" * 50)
        print("🚀 1. 空间与接口规范测试开始")
        print("=" * 50)

        obs, info = self.env.reset()
        print(f"✅ 状态空间维度: {self.env.observation_space.shape}")
        print(f"✅ 动作空间维度: {self.env.action_space.shape}")

        # 检查初始观测值是否在 [-1, 1] 范围内
        obs_min, obs_max = np.min(obs), np.max(obs)
        print(f"✅ 初始观测值范围: [{obs_min:.4f}, {obs_max:.4f}]")
        assert self.env.observation_space.contains(obs), "观测值超出 Box 定义范围！"

        # 测试 Step 无 Bug
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = self.env.step(action)
        print(f"✅ 单步交互成功！")
        print(f"   - 动作输入: {action.round(3)}")
        print(f"   - 返回奖励: {reward:.4f}")
        print(f"   - 终止状态: {terminated}, 截断状态: {truncated}")
        print(f"   - Info 键值: {list(step_info.keys())}\n")

    def run_scenarios_and_analyze(self, num_scenarios=10, steps_per_scenario=3000):
        """2, 3 & 4. 场景可视化、策略压测与 Pandas 数据分析"""
        print("=" * 50)
        print(f"🗺️ 2 & 3. 正在生成 {num_scenarios} 个场景，每个场景运行 {steps_per_scenario} 步")
        print("=" * 50)

        all_rewards_info = []

        for i in range(num_scenarios):
            # 硬重置：生成全新场景
            self.env.reset()

            # --- 可视化当前场景 ---
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(121, projection='3d')
            self._draw_3d_scene(ax1, f"Scenario {i + 1} (3D View)")

            ax2 = fig.add_subplot(122)
            self._draw_2d_scene(ax2, f"Scenario {i + 1} (2D Top-down View)")

            plt.tight_layout()
            # 自动关闭图形窗口以避免阻塞测试流（如果想仔细看可以注释掉 close，保留 show）
            plt.show(block=False)
            plt.pause(1.5)
            plt.close(fig)

            # --- 当前场景内的随机策略压测 ---
            print(f"▶️ 正在测试 Scenario {i + 1}...")
            for step in range(steps_per_scenario):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)

                # 记录核心奖励组件
                all_rewards_info.append({
                    'r_prox': info.get('r_prox', 0),
                    'r_dir': info.get('r_dir', 0),
                    'r_height': info.get('r_height', 0),
                    'r_smooth': info.get('r_smooth', 0),
                    'r_term': info.get('r_term', 0),
                    'r_continuous_clipped': info.get('r_continuous_clipped', 0)
                })

                # 软重置：如果发生碰撞、到达终点或步数截断，让无人机回到起点继续飞，不改变障碍物
                if terminated or truncated:
                    self.env.pos = self.env.start_pos.copy()
                    self.env.vel = np.zeros(3)
                    self.env.steps = 0
                    self.env.last_action = np.zeros(3)

        # --- Pandas 统计分析 ---
        self._analyze_data_with_pandas(all_rewards_info)

    def _analyze_data_with_pandas(self, data_list):
        """4. 使用 Pandas 分析并输出各项统计特征"""
        print("\n" + "=" * 50)
        print("📊 4. Pandas 奖励组件大样本统计分析结果")
        print("=" * 50)

        df = pd.DataFrame(data_list)

        # 计算基础统计量和方差
        stats = df.describe(percentiles=[.25, .5, .75]).T
        stats['variance'] = df.var()

        # 严格按照要求的列序进行展示
        cols = ['mean', 'variance', 'min', '25%', '50%', '75%', 'max']
        print(stats[cols].to_string(float_format=lambda x: f"{x:8.4f}"))

        # 数值健康性自我检查
        print("\n🔍 快速体检报告:")
        if df['r_continuous_clipped'].max() > 2.0 or df['r_continuous_clipped'].min() < -2.0:
            print("  ❌ 警告：连续奖励截断失效！存在超出 [-2, 2] 范围的单步连续奖励。")
        else:
            print("  ✅ 连续奖励截断正常，被严格限制在 [-2, 2] 内。")

        if df['r_smooth'].mean() > 0:
            print("  ❌ 警告：平滑项变成了正向奖励！请检查 Config.W_SMOOTH 符号。")
        else:
            print("  ✅ 动作平滑惩罚项逻辑正确。")

    def _draw_3d_scene(self, ax, title):
        """辅助函数：3D 场景绘制"""
        ax.set_title(title)
        r = self.cfg.OBS_RADIUS
        for pos, obs_type in self.env.obstacles:
            if obs_type == 'sphere':
                u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:10j]
                x = pos[0] + r * np.cos(u) * np.sin(v)
                y = pos[1] + r * np.sin(u) * np.sin(v)
                z = pos[2] + r * np.cos(v)
                ax.plot_wireframe(x, y, z, color='red', alpha=0.15)
            elif obs_type == 'cube':
                xx, yy, zz = pos
                d = r
                # 绘制正方体的 12 条边
                for i in [-d, d]:
                    for j in [-d, d]:
                        ax.plot([xx + i, xx + i], [yy + j, yy + j], [zz - d, zz + d], color='orange', alpha=0.3, lw=2)
                        ax.plot([xx + i, xx + i], [yy - d, yy + d], [zz + j, zz + j], color='orange', alpha=0.3, lw=2)
                        ax.plot([xx - d, xx + d], [yy + i, yy + i], [zz + j, zz + j], color='orange', alpha=0.3, lw=2)

        ax.scatter(self.env.start_pos[0], self.env.start_pos[1], self.env.start_pos[2], color='blue', s=100,
                   label='Start')
        ax.scatter(self.env.goal_pos[0], self.env.goal_pos[1], self.env.goal_pos[2], color='green', marker='*', s=200,
                   label='Goal')
        ax.set_xlim(0, 20);
        ax.set_ylim(0, 20);
        ax.set_zlim(0, 2.5)
        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.set_zlabel('Z')

    def _draw_2d_scene(self, ax, title):
        """辅助函数：2D 平面投影绘制"""
        ax.set_title(title)
        ax.set_aspect('equal')
        r = self.cfg.OBS_RADIUS
        for pos, obs_type in self.env.obstacles:
            if obs_type == 'sphere':
                circle = plt.Circle((pos[0], pos[1]), r, color='red', alpha=0.2)
                ax.add_patch(circle)
            elif obs_type == 'cube':
                side = 2 * r
                corner_x = pos[0] - r
                corner_y = pos[1] - r
                rect = plt.Rectangle((corner_x, corner_y), side, side, color='orange', alpha=0.2)
                ax.add_patch(rect)

        ax.scatter(self.env.start_pos[0], self.env.start_pos[1], color='blue', label='Start')
        ax.scatter(self.env.goal_pos[0], self.env.goal_pos[1], color='green', marker='*', s=200, label='Goal')
        ax.set_xlim(0, 20);
        ax.set_ylim(0, 20)
        ax.set_xlabel('X');
        ax.set_ylabel('Y')
        ax.grid(True, linestyle='--', alpha=0.5)


# ================= 主执行入口 =================
if __name__ == "__main__":
    tester = EnvTester()
    tester.test_spaces_and_step()
    tester.run_scenarios_and_analyze(num_scenarios=3, steps_per_scenario=3000)