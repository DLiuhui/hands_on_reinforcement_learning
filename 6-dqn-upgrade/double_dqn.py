from rl_utils import ReplayBuffer, moving_average
import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DoubleDQN:
    ''' DoubleDQN 算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon,
                 target_update, device):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # state dim as input dim
        # action dim as output dim
        self.q_net = Qnet(self.state_dim, hidden_dim, self.action_dim).to(device)
        # 目标网络
        self.target_q_net = Qnet(self.state_dim, hidden_dim, self.action_dim).to(device)
        # 优化器
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # 实现上的细节，将state tensor化后放到和net一个device上
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            # 走前向传播拿action
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q 值

        # Double Dqn 与 DQN的差异在此
        # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # DQN 取每个 state 对应的最大Q值
        # 下个状态的最大Q值
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)  # DoubleDQN 取每个 state 对应的最大 Q 值的索引, 即 action
        # 基于 max_action, 用 target-q-net 计算每个动作的Q值
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)

        q_targets = rewards + \
                    self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1

def main():
    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name, render_mode="None")
    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    # Pendulum 动作时 [-2,2] 的连续分布, 教程中把动作分解为了离散的个动作
    action_dim = 11  # 将连续动作分成 21 个离散动作 -2.0 -1.6 -1.2 ... 1.2 1.6 2.0

    def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
        action_lowbound = env.action_space.low[0]  # 连续动作的最小值
        action_upbound = env.action_space.high[0]  # 连续动作的最大值
        return action_lowbound + (discrete_action /
                                (action_dim - 1)) * (action_upbound -
                                                    action_lowbound)

    agent = DoubleDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env,
                                                   agent.action_dim)
                    next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                    done = terminated or truncated 
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Double Dqn on {}'.format(env_name))
    plt.savefig("double-dqn-return.png")

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.ylim(-200, 20)
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('Double DQN on {}'.format(env_name))
    plt.savefig("double-dqn-q-value.png")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
