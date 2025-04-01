import rl_utils
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 采用在离散动作空间上的 softmax() 函数来实现一个可学习的多项分布
        return F.softmax(self.fc2(x), dim=1)

class REINFORCE:
    ''' REINFORCE 算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # state dim as input dim
        # action dim as output dim
        self.policy_net = PolicyNet(self.state_dim, hidden_dim, self.action_dim).to(device)

        # 优化器
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):
        # 由于REINFORCE是基于策略的算法
        # 选择动作时, 不采用e-greedy, 而是根据动作概率分布随机采样

        # 实现上的细节，将 state tensor 化后放到和net一个device上
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        prob = self.policy_net(state)
        # torch.distributions.Categorical 根据传入的概率分布, 创建随机采样器
        action_dist = torch.distributions.Categorical(prob)
        # 使用随机采样器进行采样, 采样出的 index 对应的就是离散的动作
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        # 更新时, 传入一条采样轨迹
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            # 此处取策略回报的负数作为损失函数, 梯度下降最小化损失函数, 就是在最大化策略回报函数
            loss = -log_prob * G  # 计算每一步的损失函数
            loss.backward()  # 反传计算梯度
        self.optimizer.step()  # 梯度下降

def main():
    lr = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode="None")
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, lr, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
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
    plt.title('REINFORCE on {}'.format(env_name))
    plt.savefig("reinforce-returns.png")

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.savefig("reinforce-returns-moving-avg.png")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
