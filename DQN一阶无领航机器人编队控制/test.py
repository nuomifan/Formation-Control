import matplotlib.pyplot as plt
from dqn import DeepQNetwork
import numpy as np
from env import Env
import torch


def test(env, dqn, num_agent, MAX_EPISODE=1):
    score_list = []
    for i in range(num_agent):
        dqn[i].eps = 0
    env.is_render = True
    for episode in range(MAX_EPISODE):
        print("Round: ", episode)
        state = env.reset()
        while True:
            actions = [dqn[i].choose_action(state[i]) for i in range(num_follower)]
            actions.append(4)
            # 更新环境s
            next_state, reward, truncated, terminated = env.step(actions)
            state = next_state
            if truncated:
                break
    plt.clf()
    plt.cla()
    plt.plot(score_list)
    plt.show()


if __name__ == '__main__':
    num_leader = 0
    num_follower = 4
    is_render = False
    MAX_STEP = 200
    MAX_EPISODE = 50
    # 只考虑最简单的相对偏差输入，因为这样是最接近实际的，其次也降低了复杂度，并且还保持了平移不变性
    # x2-x1, y2-y1
    input_dims = 2
    # 输出为上、下、左、右不动，五种动作
    output_dims = 5
    # 隐藏层
    hidden_size = 128

    learning_rate = 1e-3
    # 奖励折扣
    gamma = 0.95
    # 探索概率
    epsilon = 0.95
    # 参数迭代频率
    replace_target_iter = 300
    # 记忆库
    memory_size = 100000
    # 采样大小
    batch_size = 32

    # 编队队形
    formation = np.array([[-1, 1],
                          [1, 1],
                          [1, -1],
                          [-1, -1]]) * 3
    # 网络拓扑
    # Adjacent matrix
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0]])

    dqn = [DeepQNetwork(input_dims=input_dims,
                        hidden_size=hidden_size,
                        output_dims=output_dims,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        epsilon=epsilon,
                        replace_target_iter=replace_target_iter,
                        memory_size=memory_size,
                        batch_size=batch_size) for i in range(num_follower)]

    # 环境
    env = Env(num_leader=num_leader,
              num_follower=num_follower,
              formation=formation,
              adjacent_matrix=A,
              is_render=is_render,
              MAX_STEP=MAX_STEP)

    for i in range(num_follower):
        dqn[i].eval_net = torch.load('models/dqn_%d.pt' % i)
    test(env, dqn, num_follower, MAX_EPISODE=10)
