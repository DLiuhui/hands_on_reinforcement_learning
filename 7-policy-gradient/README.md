# Policy-Gradient 策略梯度

* [policy-gradient 策略梯度](https://hrl.boyuai.com/chapter/2/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E7%AE%97%E6%B3%95/)
* Sutton 在2000年就提出了Policy-Gradient了, 那会神经网络都还冷着, 直到2025年Sutton终于获得了图灵奖！

## 策略梯度原理
* DQN系列是 value-based 算法, PolicyGradient 则是 policy-based 算法, 旨在学习一个目标策略, 类似与规划问题找出通过当前场景的最佳路径(策略)
* 基于策略的方法对策略进行建模(参数化), 可以采用神经网络定义一个策略的函数 $\pi_{\theta}$, 输入某个状态, 输出一个动作的概率分布。
* 目标函数: 寻找一个最优策略, 最大化策略在环境中的期望回报, $J(\theta)=E_{s_0}[V^{\pi_{\theta}}(s_0)]$
* ... 中间省略诸多证明中间过程 ... $\nabla_{\theta} J(\theta)=E_{\pi_{\theta}}[Q^{\pi_{\theta}}(s,a)\nabla_{\theta}\log{\pi_{\theta}(a|s)}]$
* 策略梯度对策略求期望, 典型的online-policy, 需要执行策略进行采样
* 梯度的更新优化策略, 让策略采样获得更高Q的动作
## REINFORCE 算法
* 基于策略梯度的范式, 采用蒙特卡洛采样法进行采样, 式子中T是与环境交互的最大步数

$\nabla_{\theta} J(\theta)=E_{\pi_{\theta}}[\sum^{T}_{t=0}(\sum^{T}_{t'=t}\gamma^{t'-t}r_{t'})\nabla_{\theta}\log{\pi_{\theta}(a_t|s_t)}]$

* 获得采样轨迹 -> 计算每个时刻t的回报 -> 更新网络参数

```python
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
            loss = -log_prob * G  # 计算每一步的损失函数
            loss.backward()  # 反传计算梯度
        self.optimizer.step()  # 梯度下降
```

## 结果
* 当MC采样的足够大, 可以得到无偏的梯度, 但是MC的估计方差较大, 导致REINFORCE输出结果不稳定(回报值的方差较大) -> 后续的Actor-Critic要解决方差大的问题
* 由于是在线算法, 需要实际进行并采样, 无法利用历史知识, 性能上可能不如 DQN类

![policy-gradient-moving-avg](reinforce-returns-moving-avg.png)
