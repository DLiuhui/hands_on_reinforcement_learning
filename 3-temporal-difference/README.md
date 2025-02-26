# 时序差分算法

* 智能体只与环境进行交互, 通过采样到的数据来学习, 这类学习方法统称为无模型的强化学习(model-free reinforcement learning)
* 时序差分算法是经典的无模型算法，也是近似求优的方法
  * 时序差分算法的增量更新方程
    * $V(s_t) \leftarrow V(s_t) + \alpha[r_t + \gamma V(s_{t+1}) - V(s_t)]$
    * $r_t + \gamma V(s_{t+1}) - V(s_t)$ 也叫 时序差分误差 TD-error
* 在线策略学习
  * 使用当前策略的采样样本进行学习, 策略更新导致样本无效
  * Sarsa
    * $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$
  * NStepSarsa 结合了MC和Sarsa的特点
    * $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma r_{t+1} + \gamma^2 r_{r+2} + ... + \gamma^n Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$
* 离线策略学习
  * 使用经验回放池将之前采样得到的样本进行再利用(运用历史数据, 这也是更接近人类的行为)
  * Q-Learning (DeepQLearning的基础了, 求近似最优解)
    * $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$
* result-env1 vs result-env2
  * Sarsa 偏保守，生成的策略会尽可能远离悬崖
  * Q-Learning 偏激进，生成的策略会尽量达到目标，会贴着悬崖走
  * env1 悬崖周围可行区域较大，Sarsa类表现较好
  * env2 设置了窄路环境，Sarsa类方法给出的策略存在犹豫不决(frozen bug!)，Q-Learning找到了相对更优的到达终点的路线
* 收敛性证明 (实在不会证了)
  * [Convergence of Q-learning: a simple proof](http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf)
