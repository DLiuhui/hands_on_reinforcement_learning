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

# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma, rewards):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解, states_num是 MDP 的状态数 '''
    # 解析解 V = (I - gamma * P)^(-1) dot R
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards)
    return value

def main():
    mdp = MDP(S, S_A, A_P, A_R, Pi_1, gamma)
    V = compute(mdp.state_prob, mdp.state_reward, gamma, mdp.state.__len__())
    print("解析解 MRP 中每个状态价值分别为\n", V)

    # 基于解析解反推, 验证解析解 V(s) = reward(s) + gamma * E[V(s')|s]
    # E[V(s')|s] = sum(P(s'|s) * V(s'))
    for i in range(mdp.state.__len__()):
        print("v{0} = {1}".format(i, gamma * np.dot(mdp.state_prob[i], V) + mdp.state_reward[i]))

if __name__ == "__main__":
    np.random.seed(0)
    main()

