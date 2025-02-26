import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cliff_walking_env import CliffWalkingEnv, print_agent

class QLearning:
    """ QLearning """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([ncol * nrow, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

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

    def update(self, s0, a0, r, s1):
        # Q-Learning update
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

def main():
    ncol = 12
    nrow = 4
    disaster = list(range(37, 47))  # env1
    # disaster = [21, 25, 26, 29, 30, 33, 45]  # env2
    target = [47]
    env = CliffWalkingEnv(ncol, nrow, disaster, target)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
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
                    agent.update(state, action, reward, next_state)
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
    plt.title('QLearning on {}'.format('Cliff Walking'))
    plt.savefig("qlearning.png")

    action_meaning = ['^', 'v', '<', '>']
    print('QLearning 算法最终收敛得到的策略为 ')
    print_agent(agent, env, action_meaning, disaster, target)

if __name__ == "__main__":
    main()
