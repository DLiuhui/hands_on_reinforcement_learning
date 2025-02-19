# 马尔可夫决策过程

* [马尔可夫决策过程](https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B/)
* RL的基础
  * 马尔可夫过程
  * MRP 马尔可夫奖励过程
    * 回报 G_{t} = R_{t} + gamma * R_{t+1} + gamma^{2} * R_{t+2}
    * 贝尔曼方程 V(s) = r(s) + gamma * sum{p(s|s')V(s')}
  * MDP 马尔可夫决策过程
    * action-value 动作价值函数 Q^{policy}(s, a)
    * state-value 状态价值函数 V^{policy}(s)
    * 通过边缘化, 将MDP降维成MRP
* MC 蒙特卡洛采样法
  * 采样 + 迭代求解
  * 老方法了, 大量的历史序列, 太消耗内存了, 而且采样过程完全没用到历史数据(知识)
  * 很显然人类的学习方式是吸取经验做反馈的, 一些类似的失败不会总是重复
* 贝尔曼最优方程
