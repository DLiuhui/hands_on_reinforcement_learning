import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cliff_walking_env import CliffWalkingEnv, print_agent

class NStepSarsa:
    """ N-Step Sarsa算法 """
    def __init__(self, n_step, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.n_step = n_step
        self.Q_table = np.zeros([ncol * nrow, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        # 历史信息记录
        self.action_list = []
        self.state_list = []
        self.reward_list = []

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)  # 随机选一个动作
        else:
            action = np.argmax(self.Q_table[state])  # 选Q(s,a)最大的
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n_step:
            G = self.Q_table[s1, a1]  # 得到Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n_step)):
                G = self.gamma * G + self.reward_list[i]
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                if done and i > 0:
                    st = self.state_list[i]
                    at = self.action_list[i]
                    self.Q_table[st, at] += self.alpha * (G - self.Q_table[st, at])
            # 弹出队头, 更新Q-table; 新的队头s, a等待下一次更新
            st = self.state_list.pop(0)
            at = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[st, at] += self.alpha * (G - self.Q_table[st, at])
        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []

def main():
    ncol = 12
    nrow = 4
    disaster = list(range(37, 47))  # env1
    # disaster = [21, 25, 26, 29, 30, 33, 45]  # env2
    target = [47]
    env = CliffWalkingEnv(ncol, nrow, disaster, target)
    np.random.seed(0)
    n_step = 5  # 5步Sarsa算法
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = NStepSarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500

    return_list = []
    for i  in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action, done)
                    state = next_state
                    action = next_action
                # episode_return 是整个探索过程的 reward 值
                # 优化目标就是最大化这个值
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
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
    plt.title('NStepSarsa on {}'.format('Cliff Walking'))
    plt.savefig("n_step_sarsa.png")

    action_meaning = ['^', 'v', '<', '>']
    print('NStepSarsa 算法最终收敛得到的策略为 ')
    print_agent(agent, env, action_meaning, disaster, target)

if __name__ == "__main__":
    main()
