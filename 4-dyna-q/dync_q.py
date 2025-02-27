from cliff_walking_env import CliffWalkingEnv, print_agent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

class DynaQ:
    """ Dyna-Q算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def take_action(self, state):  #选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        # Q(s, a) <- Q(s, a) + alpha[r + gamma * maxQ(s', a') - Q(s, a)]
        # M(s, a) <- (r, s')  更新环境模型
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1
        for _ in range(self.n_planning):
            # 随机选择曾经遇到过的状态动作对
            # 这里不需要更新动作对, 只需要从model里获取
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    disaster = list(range(37, 47))  # env1
    # disaster = [21, 25, 26, 29, 30, 33, 45]  # env2
    target = [47]
    env = CliffWalkingEnv(ncol, nrow, disaster, target)

    agent = DynaQ(
        ncol=env.ncol, nrow=env.nrow,
        epsilon=0.01,
        alpha=0.1,
        gamma=0.9,
        n_planning=n_planning)

    num_episodes = 300  # 智能体在环境中运行多少条序列
    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    print('DynaQ 算法 n-planning = {0} 最终收敛得到的策略为 '.format(n_planning))
    print_agent(agent, env, ['^', 'v', '<', '>'], disaster, target)
    return return_list

def main():
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20]  # 不同的超参数 与环境model交互执行Q-planning的次数
    for n_planning in n_planning_list:
        print('Q-planning次数为：%d' % n_planning)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list, label=str(n_planning) + ' planning steps')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.savefig('dyna_q.png')

if __name__ == "__main__":
    main()
