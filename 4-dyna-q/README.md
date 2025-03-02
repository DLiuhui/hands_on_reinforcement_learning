# 4 Dyna-Q

* [Dyna-Q算法](https://hrl.boyuai.com/chapter/1/dyna-q%E7%AE%97%E6%B3%95)
* Dyna-Q model-based 方法范式
    * model-based 方法增加了对环境的建模
    * 作为对比，Sarsa和Q-Learning不需要对环境建模，参考采样结果
    * 获取r, s'
    * $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'}Q(s', a') - Q(s, a)]$
    * $M(s,a) \leftarrow r, s'$
    * 然后执行 n-planning 次
        * 每次从 M 中随机采样一个状态和动作 $r_m, s'_m \leftarrow M(s_m, a_m)$
        * $Q(s_m, a_m) \leftarrow Q(s_m, a_m) + \alpha[r + \gamma \max_{a'}Q(s'_m, a') - Q(s_m, a_m)]$

![model-based方法](img/instruc.png)

* 例子里DynaQ的迭代思路就是DQN的思路：采样 + mem 回放
* 随着 Q-planning 步数的增多，Dyna-Q 算法的收敛速度也随之变快
* 综合来看 n-planning 为2, 在两种env下都表现较好
* 不过dp、td、dyna-q几种传统方法的环境都是偏确定性的，离散的状态与动作空间，转移也是确定性的，当s、a空间增大或连续，Q_table这种查表式的方法就不再适合了，函数模型——亦或者是神经网络模型，就更加的适合。
