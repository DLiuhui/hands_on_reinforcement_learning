import numpy as np

# 定义mdp的要素
S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
# 状态-动作集合
S_A = {
    "s1": {"s1-保持s1", "s1-前往s2"},
    "s2": {"s2-前往s1", "s2-前往s3"},
    "s3": {"s3-前往s4", "s3-前往s5"},
    "s4": {"s4-前往s5", "s4-概率前往"},
    "s5": {"s5-保持s5"}
}
# 动作-状态转移函数
A_P = {
    "s1-保持s1": {"s1": 1.0},
    "s1-前往s2": {"s2": 1.0},
    "s2-前往s1": {"s1": 1.0},
    "s2-前往s3": {"s3": 1.0},
    "s3-前往s4": {"s4": 1.0},
    "s3-前往s5": {"s5": 1.0},
    "s4-前往s5": {"s5": 1.0},
    "s4-概率前往": {"s2": 0.2, "s3": 0.4, "s4": 0.4},
    "s5-保持s5": {"s5": 1.0}
}
# 奖励函数
A_R = {
    "s1-保持s1": -1.0,
    "s1-前往s2": 0.0,
    "s2-前往s1": -1.0,
    "s2-前往s3": -2.0,
    "s3-前往s4": -2.0,
    "s3-前往s5": 0.0,
    "s4-前往s5": 10.0,
    "s4-概率前往": 1.0,
    "s5-保持s5": 0.0
}
gamma = 0.5  # 折扣因子

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
    "s5-保持s5": 1.0
}

# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
    "s5-保持s5": 1.0
}

class MDP:
    def __init__(self, state, state_actions, action_probs, action_reward, policy, gamma):
        self.state = state
        self.state_actions = state_actions
        self.action_probs = action_probs
        self.action_reward = action_reward
        self.policy = policy
        self.gamma = gamma
        # 解算 MDP 边缘化 aciton_prob 和 action_reward -> MRP
        # 计算 state 转移概率, 对每个 state 的 action 进行积分
        self.state_prob = np.zeros((len(self.state), len(self.state)))
        for s in self.state:
            actions = self.state_actions.get(s, None)
            if actions is None:
                continue
            for a in actions:
                probs = self.action_probs.get(a, None)
                if probs is None:
                    continue
                for s_next in probs:
                    # print(a, self.policy.get(a, 0.0))
                    # print(s_next, probs[s_next])
                    self.state_prob[self.state.index(s)][self.state.index(s_next)] += \
                        probs[s_next] * self.policy.get(a, 0.0)
        print("state 状态转移矩阵\n", self.state_prob)
        # 计算 state 奖励
        # 对每个状态下的 action_reward 进行积分
        self.state_reward = np.zeros((len(self.state), 1))
        for s in self.state:
            actions = self.state_actions.get(s, None)
            if actions is None:
                continue
            for a in actions:
                self.state_reward[self.state.index(s)] += \
                    self.action_reward.get(a, 0.0) * self.policy.get(a, 0.0)
        print("state value 向量\n", self.state_reward)

def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解, states_num是 MDP 的状态数 '''
    # 解析解 V = (I - gamma * P)^(-1) dot R
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards)
    return value

def sample(mdp: MDP, timestep_max, number):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = mdp.state[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp_prob = np.random.rand(), 0
            # 在状态s下根据策略随机选择动作
            actions = mdp.state_actions[s]
            for a_opt in actions:
                temp_prob += mdp.policy.get(a_opt, 0)
                if temp_prob > rand:
                    a = a_opt
                    r = mdp.action_reward.get(a_opt, 0)
                    break
            rand, temp_prob = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            next_states = mdp.action_probs[a]
            for s_opt in next_states:
                temp_prob += next_states[s_opt]
                if temp_prob > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes

def MC(episodes, V, N, gamma):
    ''' 蒙特卡洛方法求解MDP,episodes是采样得到的序列,V是初始状态价值,N是状态的个数 '''
    for episode in episodes:
        G = 0
        for s, a, r, s_next in reversed(episode):
            G = gamma * G + r
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]

def main():
    mdp = MDP(S, S_A, A_P, A_R, Pi_1, gamma)
    V = compute(mdp.state_prob, mdp.state_reward, gamma, mdp.state.__len__())
    print("解析解 MRP 中每个状态价值分别为\n", V)

    # mc
    # 采样5次,每个序列最长不超过20步
    n_samples = 10000
    n_steps = 20
    episodes = sample(mdp, n_steps, n_samples)
    # for i in range(n_samples):
    #     print('第{0}条序列\n{1}'.format(i + 1, episodes[i]))
    V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    MC(episodes, V, N, gamma)
    print("蒙特卡洛方法求解MDP, 策略Pi_1, 采样次数{0}, 每次采样最大步数{1}, 采样得到的状态价值为\n{2}".format(
          n_samples, n_steps, V))

if __name__ == "__main__":
    np.random.seed(0)
    main()

