# 多臂老虎机

* 入门课 [多臂老虎机](https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA)
* 老问题了, 展现了exploration & exploitation 问题的解决范式, 迭代求解, 平衡探索与挖掘, 价值函数等等
* 多臂老虎机模型 <A, R>
    * $Q(a) = E_{r\sim R(.|a)}[r]$  动作的期望价值, 奖励的期望
    * 最优激励 $Q^{*}=max_{a\in A}Q(a)$
    * $Regret(a) = Q^{*} - Q(a)$, 损失函数, 最优期望与实际之差
    * 对于$a\in A$, 该动作的计数 $N(a)=0$ 该动作的价值估计 $\hat{Q}(a)=0$
* 采样: 根据e-greedy等方法采样一个动作a, 并获取回报r
* 迭代更新
    * $N(a) = N(a) + 1$
    * $\hat{Q}(a) = \hat{Q}(a) + \frac{1}{N(a)} * [r - \hat{Q}(a)]$
