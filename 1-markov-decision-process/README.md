# 马尔可夫决策过程

* [马尔可夫决策过程](https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B/)
* $最优策略 = argmax_{policy}E(s,a) \sim 策略占用度量[Reward(s,a)]$
* 马尔可夫决策过程是RL的基础

## MRP 马尔可夫奖励过程
* 要素
  * S 有限状态集
  * P 状态转移矩阵
  * r 到每个状态后的奖励
  * gamma 衰减值, 平衡关注当下价值与关注长期价值
* 回报 $G_{t} = R_{t} + \gamma R_{t+1} + \gamma^{2} R_{t+2} + ... = \sum_{k=0}^{\infty}\gamma^{k} R_{t+k}$  类似与计算现金折旧
* 价值函数
  * $V(s) = E[G_t | S_t = s] = E[R_t + \gamma G_{t+1} | S_t = s]$
  * $V(s) = E[R_t | S_t = s] + \gamma E[V(S_{t+1})|S_t = s]$
* **贝尔曼方程** $V(s) = r(s) + \gamma \sum_{s'\in S} P(s'|s)V(s')$
  * 其中$r(s) = E[R_t | S_t = s]$就是s处的回报
  * $p(s|s')$是状态转移概率
* 写成矩阵式 $V = R + \gamma PV$, 可以求解出 $V = (I - \gamma P)^{-1}R$
## MDP 马尔可夫决策过程
* 要素
  * MRP的基础上, 引入动作action A(离散或连续)
  * 奖励 r(s,a)
  * 转移概率 P(s'|s,a)
  * policy 策略 $\pi(a|s) = P(A_t=a|S_t=s)$, 表示处于状态S_t时, 执行每个动作的概率, **策略可以是离散的概率, 也可以是一个概率分布函数**
* action-value 动作价值函数
  * $Q^{\pi}(s, a) = E_{\pi}[G_t | S_t = s, A_t = a]$
  * $Q^{\pi}(s, a) = r(s,a) + \gamma \sum_{s'\in S}P(s'|s,a)V^{\pi}(s')$
* state-value 状态价值函数
  * $V^{\pi}(s) = E_{\pi}[G_t | S_t = s]$
  * $V^{\pi}(s) = \sum_{a\in A}\pi(a|s)Q^{\pi}(s, a)$
---
### 贝尔曼期望方程
* $V^{\pi}(s) = \sum_{a\in A}\pi(a|s)r(s,a) + \gamma \sum_{a\in A}\pi(a|s) \sum_{s'\in S}P(s'|s,a)V^{\pi}(s')$
* $Q^{\pi}(s, a) = r(s,a) + \gamma \sum_{s'\in S}P(s'|s,a)\sum_{a'\in A}\pi(a'|s')Q^{\pi}(s', a')$
---
* 降维 MDP -> MRP
  * $r(s) = \sum_{a\in A}\pi(a|s)r(s,a)$ 对动作积分, 得到state的奖励
  * $P(s'|s) = \sum_{a\in A}\pi(a|s)\sum_{s'\in S}P(s'|s,a)$  对动作积分, 得到状态转移概率
  * 这样V就可以降维到与action无关, 参考MRP求解可得到解析解
* MC 蒙特卡洛采样法
  * 采样 + 迭代求解
    * $V^{\pi}(s) = \frac{1}{N} \sum_{i=1}^{N} G_t^{i}$
  * 递推式
    * $N(s) = N(s) + 1$
    * $R(s) = R(s) + G_t$
    * $V(s) = R(s) / N(s)$
    * 对于s的新采样, 有 $V(s) = V(s) + \frac{1}{N(s)} (G_{t}^{N} - V(s))$, G与s无关, G是一轮采样序列的回报
  * 非常经典的方法了
  * 缺点也很明显: 需要大量的历史序列, 太消耗内存了; 迭代过程完全没用到更早采样的历史数据(知识); 虽然是无偏估计但是高方差波动
  * 很显然人类的学习方式是吸取经验做反馈的, 一些类似的失败不会总是重复
* 贝尔曼最优方程 $\pi^{*}(s)$
  * 最优状态价值 $V^{*}(s) = max_{\pi}V^{\pi}(s)$
  * 最优动作价值 $Q^{*}(s,a) = max_{\pi}Q^{\pi}(s,a)$
  * 最优动作 -> 组成最优策略 -> 运行得到最优状态
  * 所以在优化时, 可以遵循贝尔曼最优方程
---  
### 贝尔曼最优方程
* $V^{*}(s) = max_{a\in A}\{r(s,a) + \gamma \sum_{s'\in S}P(s'|s,a)V^{*}(s')\}$
* $Q^{*}(s, a) = r(s,a) + \gamma \sum_{s'\in S}P(s'|s,a)max_{a'\in A}Q^{*}(s', a')$
* 不同于贝尔曼方程要关注策略$\pi(a|s)$, 最优策略下, 可以选出确定的动作, 即让V和Q最大化的action
---
* 在有限的state和action下, 可以通过动态规划DP来求解最优策略和最优状态价值
* 而在无限的state和action下, 就需要通过建模与优化来逼近最优或局部最优解了
