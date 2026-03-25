import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

from env import UAVEnv, Config, Plot


# ==========================================
# 1. 神经网络结构定义 (Actor-Critic)
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """网络权重正交初始化"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Critic 网络：评估当前状态的 Value (输出标量)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

        # Actor 网络：输出动作的高斯分布均值 (Mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01)  # 初始化方差极小，让初期动作输出在0附近
        )

        # 动作的对数标准差 (独立于状态的可学习参数)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# ==========================================
# 2. PPO 训练器核心逻辑
# ==========================================
class PPOTrainer:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 训练设备: {self.device}")

        # --- PPO 核心超参数 ---
        self.learning_rate = 3e-4
        self.total_timesteps = 1200000  # 建议训练120万步
        self.batch_size = 2048  # 每次收集经验步数
        self.minibatch_size = 256  # 每次梯度更新的批大小
        self.update_epochs = 10  # 每次收集后网络更新轮数

        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE 优势估计衰减参数
        self.clip_coef = 0.2  # PPO 截断范围
        self.ent_coef = 0.01  # 熵奖励系数 (鼓励探索)
        self.vf_coef = 0.5  # Value 损失系数
        self.max_grad_norm = 0.5  # 梯度裁剪最大范数

        # --- 实例化网络与优化器 ---
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.agent = Agent(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # --- 统计与日志 ---
        self.stats = {
            "reward": [], "length": [], "loss": [],
            "success": [], "min_dist": []
        }
        self.save_dir = "./models"
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        """主训练循环"""
        # 初始化存储 Buffer
        obs_buf = torch.zeros((self.batch_size, self.env.observation_space.shape[0])).to(self.device)
        act_buf = torch.zeros((self.batch_size, self.env.action_space.shape[0])).to(self.device)
        logp_buf = torch.zeros(self.batch_size).to(self.device)
        rew_buf = torch.zeros(self.batch_size).to(self.device)
        done_buf = torch.zeros(self.batch_size).to(self.device)
        val_buf = torch.zeros(self.batch_size).to(self.device)

        global_step = 0
        start_time = time.time()
        obs, _ = self.env.reset()

        # 回合过程追踪变量
        ep_reward = 0
        ep_len = 0
        ep_min_dist = float('inf')

        # 计算总的迭代次数
        num_updates = self.total_timesteps // self.batch_size

        print("\n" + "=" * 80)
        print(
            f"{'Step':>8} | {'Avg_Rew':>8} | {'Avg_Len':>7} | {'Avg_Loss':>8} | {'Succ_Rate':>9} | {'Min_Dist':>8} | {'FPS':>5}")
        print("=" * 80)

        for update in range(1, num_updates + 1):
            # 1. 学习率线性衰减
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

            # 2. 收集经验 Rollout
            for step in range(self.batch_size):
                global_step += 1
                obs_buf[step] = torch.tensor(obs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(obs_buf[step].unsqueeze(0))
                    val_buf[step] = value.flatten()
                act_buf[step] = action.flatten()
                logp_buf[step] = logprob.flatten()

                # 执行动作并传入环境 (限幅以符合物理边界)
                env_action = action.cpu().numpy().flatten()
                env_action = np.clip(env_action, -1.0, 1.0)

                next_obs, reward, terminated, truncated, info = self.env.step(env_action)

                rew_buf[step] = torch.tensor(reward).to(self.device)
                done_buf[step] = torch.tensor(terminated or truncated).to(self.device)

                # 更新回合追踪状态
                ep_reward += reward
                ep_len += 1
                ep_min_dist = min(ep_min_dist, info['dist'])
                obs = next_obs

                if terminated or truncated:
                    self.stats['reward'].append(ep_reward)
                    self.stats['length'].append(ep_len)
                    self.stats['min_dist'].append(ep_min_dist)

                    # 判定是否成功到达终点
                    is_success = 1.0 if (info['dist'] < 0.5 and self.env.pos[2] < 0.1) else 0.0
                    self.stats['success'].append(is_success)

                    obs, _ = self.env.reset()
                    ep_reward, ep_len, ep_min_dist = 0, 0, float('inf')

            # 3. 计算优势估计 GAE
            with torch.no_grad():
                next_value = self.agent.get_value(
                    torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)).flatten()
                advantages = torch.zeros_like(rew_buf).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.batch_size)):
                    if t == self.batch_size - 1:
                        nextnonterminal = 1.0 - (terminated or truncated)
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t]
                        nextvalues = val_buf[t + 1]
                    delta = rew_buf[t] + self.gamma * nextvalues * nextnonterminal - val_buf[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + val_buf

            # 4. PPO 核心网络更新 (Mini-batch)
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            update_loss = []

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(obs_buf[mb_inds],
                                                                                       act_buf[mb_inds])
                    logratio = newlogprob - logp_buf[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = advantages[mb_inds]
                    # 优势函数归一化 (Batch 级稳定性提升)
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((newvalue.flatten() - returns[mb_inds]) ** 2).mean()

                    # Entropy Loss (鼓励探索)
                    entropy_loss = entropy.mean()

                    # 综合 Loss
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)  # 梯度裁剪
                    self.optimizer.step()

                    update_loss.append(loss.item())

            # 保存平均 loss
            self.stats['loss'].append(np.mean(update_loss))

            # 5. 定期日志输出
            if update % 5 == 0 or update == 1:
                fps = int(global_step / (time.time() - start_time))
                avg_r = np.mean(self.stats['reward'][-50:]) if len(self.stats['reward']) > 0 else 0
                avg_l = np.mean(self.stats['length'][-50:]) if len(self.stats['length']) > 0 else 0
                avg_s = np.mean(self.stats['success'][-50:]) if len(self.stats['success']) > 0 else 0
                avg_d = np.mean(self.stats['min_dist'][-50:]) if len(self.stats['min_dist']) > 0 else 20.0
                curr_loss = np.mean(update_loss)

                print(
                    f"{global_step:8d} | {avg_r:8.2f} | {avg_l:7.0f} | {curr_loss:8.4f} | {avg_s:8.1%} | {avg_d:7.2f}m | {fps:5d}")

            # 6. 模型保存 (每 25 轮 Update 保存一次，约等于 51200 步)
            if update % 25 == 0:
               save_path = os.path.join(self.save_dir, f"ppo_uav_{global_step}.pth")
               torch.save(self.agent.state_dict(), save_path)
               print(f"💾 模型已保存至: {save_path}")

        print("\n🎉 训练完全结束！")


# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 实例化环境与训练器
    env = UAVEnv()
    trainer = PPOTrainer(env)

    try:
        # 开始训练
        trainer.train()
    except KeyboardInterrupt:
        print("\n⏸️ 检测到手动中断，正在整理当前训练数据...")

    # 训练结束后，调用 env.py 中的通用绘图接口
    print("📈 正在生成训练数据曲线图...")
    Plot.plot_training_curves(trainer.stats, save_path="training_final_curves.png")