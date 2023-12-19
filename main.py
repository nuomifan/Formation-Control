import torch
from env import Env
import numpy as np
from tqdm import trange
from sac import SoftActorCritic
from dqn import DeepQNetwork

if __name__ == '__main__':
    num_leader = 1
    num_follower = 4
    is_render = False
    MAX_STEP = 500
    MAX_EPISODE = 50
    # 只考虑最简单的相对偏差输入，因为这样是最接近实际的，其次也降低了复杂度，并且还保持了平移不变性
    # x2-x1, y2-y1, vx2-vx1, vy2-vy1
    input_dims = 4
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
                          [-1, -1],
                          [0, 0]]) * 3
    # 网络拓扑
    # Adjacent matrix
    A = np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])

    agent = [DeepQNetwork(input_dims=input_dims,
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

    step = 0

    frames = 0

    score_list = []
    for episode in trange(MAX_EPISODE):
        state = env.reset()
        score = 0
        while True:
            # 选择动作
            actions = [agent[i].choose_action(state[i]) for i in range(num_follower)]
            actions.append(0)
            # actions[0] = agent.choose_action(state[0])
            # 更新环境s
            next_state, reward, truncated, terminated = env.step(actions)
            # 存储记忆
            for i in range(num_follower):
                agent[i].store_transition(state[i], actions[i], reward[i], next_state[i], terminated[i])
            # agent.store_transition(state[0], actions[0], reward[0], next_state[0], terminated[0])
            score += reward[0]
            state = next_state
            if truncated:
                break
            frames += 1
            if frames > 1000:
                for i in range(num_follower):
                    agent[i].learn()
                # agent.learn()
        if score > 100:
            env.show()
        print(score)
    # for i in range(num_follower):
    #     torch.save(dqn[i].eval_net, 'models/dqn_%d.pt' % i)
