import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Config 参数类：集中管理所有超参数
# ==========================================
class Config:
    # --- 空间与物理参数 ---
    WORLD_SIZE = 20.0
    MAX_HEIGHT = 2.0
    TARGET_HEIGHT = 1.0
    MAX_VEL_XY = 1.5
    MAX_VEL_Z = 0.8
    DT = 0.1
    MAX_STEPS = 500

    # --- 障碍物与目标设置 ---
    OBS_COUNT = 5
    OBS_RADIUS = 1.0
    MIN_DIST_OBS = 2.5
    MIN_DIST_ST_GL = 2.0
    MIN_ST_GL_DIST = 15.0

    # --- 传感器参数 ---
    LIDAR_RAYS = 16
    LIDAR_RANGE = 5.0

    # --- 奖励系数 (严格标定版) ---
    W_STEP = -0.4  # 每步扣血，逼迫快速结束
    W_HEIGHT = 0.2  # 鼓励保持高度1m飞行
    W_DIR = 0.2  # 鼓励朝向终点飞行
    W_PROX_FAR = 3.0  # 鼓励靠近终点
    W_PROX_NEAR = 6.0  # 靠近终点系数更大
    W_SMOOTH = 0.05  # 鼓励平滑

    # --- 事件/终端奖励 ---
    R_COLLISION = -200.0
    R_GOAL = 200.0


# ==========================================
# 2. Env 环境类：遵循 Gymnasium 标准接口
# ==========================================
class UAVEnv(gym.Env):
    """三维无人机路径规划环境"""

    def __init__(self):
        super(UAVEnv, self).__init__()
        self.cfg = Config()

        # 动作空间：三维速度 [vx, vy, vz]，输入范围 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 状态空间维度：自身归一化位置(3) + 归一化速度(3) + 相对目标向量(3) + 归一化距离(1) + 雷达(16) = 26维
        obs_dim = 3 + 3 + 3 + 1 + self.cfg.LIDAR_RAYS
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.pos = None
        self.vel = None
        self.start_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.steps = 0
        self.last_action = np.zeros(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = np.zeros(3)

        # 1. 随机生成符合间距约束的起终点 (高度 z=0)
        while True:
            self.start_pos = np.array([np.random.uniform(2, 18), np.random.uniform(2, 18), 0.0])
            self.goal_pos = np.array([np.random.uniform(2, 18), np.random.uniform(2, 18), 0.0])
            if np.linalg.norm(self.start_pos - self.goal_pos) >= self.cfg.MIN_ST_GL_DIST:
                break

        # 2. 拒绝采样法生成障碍物 (包含正方体和球体)
        self.obstacles = []
        while len(self.obstacles) < self.cfg.OBS_COUNT:
            obs_type = np.random.choice(['sphere', 'cube'])
            obs_pos = np.array([np.random.uniform(2, 18), np.random.uniform(2, 18), np.random.uniform(0.5, 1.5)])

            valid = True
            for other_pos, _ in self.obstacles:
                if np.linalg.norm(obs_pos - other_pos) < self.cfg.MIN_DIST_OBS: valid = False
            if np.linalg.norm(obs_pos - self.start_pos) < self.cfg.MIN_DIST_ST_GL + self.cfg.OBS_RADIUS: valid = False
            if np.linalg.norm(obs_pos - self.goal_pos) < self.cfg.MIN_DIST_ST_GL + self.cfg.OBS_RADIUS: valid = False

            if valid: self.obstacles.append((obs_pos, obs_type))

        self.pos = self.start_pos.copy()
        self.vel = np.zeros(3)

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        # 1. 动作解映射与运动学更新 (质点模型)
        target_vel = np.array([
            action[0] * self.cfg.MAX_VEL_XY,
            action[1] * self.cfg.MAX_VEL_XY,
            action[2] * self.cfg.MAX_VEL_Z
        ])
        old_pos = self.pos.copy()
        self.pos += target_vel * self.cfg.DT
        self.vel = target_vel

        # 2. 感知与碰撞检测
        lidar_data = self._get_lidar()
        collision = np.any(lidar_data < 0.1)
        out_of_bounds = not (0 <= self.pos[0] <= 20 and 0 <= self.pos[1] <= 20 and 0 <= self.pos[2] <= 2)
        fail = collision or out_of_bounds

        # 3. 目标达成判定
        dist_to_goal = np.linalg.norm(self.pos - self.goal_pos)
        success = (dist_to_goal < 0.5) and (self.pos[2] < 0.1)

        # 4. 计算奖励 (核心约束: 连续奖励截断与事件奖励分离)
        reward, reward_info = self._compute_reward(action, old_pos, dist_to_goal, fail, success)

        # 5. 状态机标志位
        terminated = bool(fail or success)
        truncated = bool(self.steps >= self.cfg.MAX_STEPS)
        self.last_action = action.copy()

        # 补充距离信息用于外部统计
        reward_info['dist'] = dist_to_goal

        return self._get_obs(), reward, terminated, truncated, reward_info

    def _get_obs(self):
        """状态空间归一化处理至 [-1, 1] 范围内"""
        pos_norm = (self.pos / np.array([10, 10, 1])) - 1.0
        vel_norm = self.vel / np.array([self.cfg.MAX_VEL_XY, self.cfg.MAX_VEL_XY, self.cfg.MAX_VEL_Z])
        rel_goal = (self.goal_pos - self.pos) / self.cfg.WORLD_SIZE
        dist_norm = np.clip(np.linalg.norm(self.goal_pos - self.pos) / self.cfg.WORLD_SIZE, 0, 1)
        lidar_norm = self._get_lidar() / self.cfg.LIDAR_RANGE

        obs = np.concatenate([pos_norm, vel_norm, rel_goal, [dist_norm], lidar_norm])
        return obs.astype(np.float32)

    def _get_lidar(self):
        """模拟水平多线局部雷达感知障碍物距离"""
        scan = np.ones(self.cfg.LIDAR_RAYS) * self.cfg.LIDAR_RANGE
        angles = np.linspace(0, 2 * np.pi, self.cfg.LIDAR_RAYS, endpoint=False)
        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle), 0])
            for obs_pos, obs_type in self.obstacles:
                rel_pos = obs_pos - self.pos
                dist_proj = np.dot(rel_pos, ray_dir)
                if dist_proj > 0:
                    # 到射线向量的垂直距离判断
                    dist_perp = np.linalg.norm(rel_pos - dist_proj * ray_dir)
                    if dist_perp < self.cfg.OBS_RADIUS:
                        scan[i] = min(scan[i], dist_proj)
        return scan

    def _compute_reward(self, action, old_pos, dist_to_goal, fail, success):
        # 1. 靠近奖励
        dist_prev = np.linalg.norm(old_pos - self.goal_pos)
        prox_coeff = self.cfg.W_PROX_NEAR if dist_to_goal < 5.0 else self.cfg.W_PROX_FAR
        r_prox = (dist_prev - dist_to_goal) * prox_coeff

        # 2. 方向奖励 (与水平速度成正比，速度为0则无方向奖励，杜绝原地刷分)
        speed_xy = np.linalg.norm(self.vel[:2])
        speed_ratio = np.clip(speed_xy / self.cfg.MAX_VEL_XY, 0.0, 1.0)
        goal_dir = (self.goal_pos - self.pos) / (dist_to_goal + 1e-8)
        vel_dir = self.vel / (np.linalg.norm(self.vel) + 1e-8)
        r_dir = np.dot(goal_dir, vel_dir) * self.cfg.W_DIR * speed_ratio

        dist_xy = np.linalg.norm(self.pos[:2] - self.goal_pos[:2])  # 水平面距离

        if dist_xy > 2.0:
            # 2m 以外，目标高度恒定为 TARGET_HEIGHT (1.0)
            target_z = self.cfg.TARGET_HEIGHT
        else:
            # 2m 以内，目标高度随着距离线性降低（平滑无断崖）
            target_z = (dist_xy / 2.0) * self.cfg.TARGET_HEIGHT

        # 统一计算高度偏差得分
        height_score = 1.0 - abs(self.pos[2] - target_z)
        r_height = height_score * self.cfg.W_HEIGHT

        # 在 2m 内，附加降落姿态惩罚
        if dist_xy <= 2.0:
            if self.vel[2] < -0.5:  # 惩罚下降太快“砸”地
                r_height -= abs(self.vel[2] + 0.5) * self.cfg.W_HEIGHT
            elif self.vel[2] > 0.0:  # 惩罚降落区往上飞
                r_height -= self.vel[2] * self.cfg.W_HEIGHT

        # 4. 平滑惩罚
        r_smooth = -np.sum((action - self.last_action) ** 2) * self.cfg.W_SMOOTH

        # --- 截断连续奖励组件，依然死守 [-2.0, 2.0] ---
        continuous_reward = self.cfg.W_STEP + r_prox + r_dir + r_height + r_smooth
        continuous_reward = np.clip(continuous_reward, -2.0, 2.0)

        # 5. 事件奖励
        r_term = 0.0
        if success:
            r_term = self.cfg.R_GOAL
        elif fail:
            r_term = self.cfg.R_COLLISION

        total_reward = continuous_reward + r_term

        info = {
            "r_step": self.cfg.W_STEP,
            "r_prox": r_prox,
            "r_dir": r_dir,
            "r_height": r_height,
            "r_smooth": r_smooth,
            "r_term": r_term,
            "r_continuous_clipped": continuous_reward,
            "dist": dist_to_goal
        }
        return total_reward, info

# ==========================================
# 3. Plot 绘图类：通用的数据可视化接口
# ==========================================
class Plot:

    @staticmethod
    def plot_training_curves(stats_dict, save_path="training_curves.png", window=20):
        """
        接收训练过程中的统计字典，绘制多维度趋势图。
        参数:
            stats_dict (dict): 形如 {'reward': [...], 'success': [...], 'dist': [...]}
            save_path (str): 图像保存路径
            window (int): 移动平均窗口大小
        """
        # 过滤掉空的列表
        valid_keys = [k for k in stats_dict.keys() if isinstance(stats_dict[k], list) and len(stats_dict[k]) > 0]
        num_plots = len(valid_keys)

        if num_plots == 0:
            print("⚠️ 警告：传入的数据字典为空，无法绘图。")
            return

        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4.5))
        if num_plots == 1:
            axes = [axes]

        for i, key in enumerate(valid_keys):
            ax = axes[i]
            data = stats_dict[key]

            # 绘制原始数据的浅色背景线
            ax.plot(data, color='gray', alpha=0.3, label='Raw')

            # 绘制平滑处理后的趋势线
            if len(data) >= window:
                smooth_data = pd.Series(data).rolling(window=window, min_periods=1).mean()
                ax.plot(smooth_data, color='blue', lw=2, label=f'MA({window})')

            ax.set_title(f"Metric: {key.capitalize()}")
            ax.set_xlabel("Epoch / Episode")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"📊 训练曲线图已成功保存至: {save_path}")
        plt.show()
