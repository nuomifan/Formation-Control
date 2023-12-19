import numpy as np
import matplotlib.pyplot as plt


class Env:
    def __init__(self, num_leader,
                 num_follower,
                 formation,
                 adjacent_matrix,
                 is_render=True,
                 MAX_STEP=500):
        self._num_agent = num_leader + num_follower
        self._num_followers = num_follower
        self._num_leader = num_leader
        self.MAX_STEP = MAX_STEP
        # 加速度 0.01 m/s^2
        self.acc = np.zeros(shape=(self._num_agent, 2))
        self.vel = np.zeros(shape=(self._num_agent, 2))
        self.pos = np.zeros(shape=(self._num_agent, 2))

        # 目标队形
        self.formation = formation
        # 邻接矩阵
        self.A = adjacent_matrix
        # 拉式矩阵
        self.L = self.A
        for i in range(self._num_agent):
            self.L[i, i] = -sum(self.L[i, :])
        self.L = np.kron(self.L, np.eye(2))

        self.is_render = is_render
        self.step_ctr = 0

        self._state_dim = self._num_agent * 6
        self._action_dim = 5

        self.action = np.array([[1, 0],
                                [0, 1],
                                [-1, 0],
                                [0, -1],
                                [0, 0]]) * 0.1

    def state_space(self):
        return self._state_dim

    def action_space(self):
        return self._action_dim

    def reset(self):
        # 将机器人移动到随机位置
        # self.pos = np.round(np.random.rand(self._num_agent, 2) * 10 - 5, 1)
        self.pos = np.array([[-3.8, -3.2],
                             [-2.9, -2.3],
                             [-1.9, -4.1],
                             [0.2, 4.1]])
        self.acc = np.zeros(shape=(self._num_agent, 2))
        self.vel = np.zeros(shape=(self._num_agent, 2))

        # 计算相对偏移量
        current_pos = self.pos.flatten()
        state = np.dot(self.L, current_pos).reshape(self._num_agent, 2)
        self.step_ctr = 0
        return state

    def action_sample(self):
        actions = np.random.rand(0, 5, self._num_agent)
        return actions

    def calculate_reward(self, current_pos, next_pos):
        terminated = []
        reward = []
        # 以相对偏移方式计算
        # 考虑别的机器人的运动
        current_pos = current_pos.flatten()
        next_pos = next_pos.flatten()
        previous_relative_distance = np.dot(self.L, current_pos)
        current_relative_distance = np.dot(self.L, next_pos)
        target_relative_distance = np.dot(self.L, self.formation.flatten())

        for i in range(self._num_agent):
            previous_distance = np.linalg.norm(previous_relative_distance[2 * i:2 * (i + 1)] -
                                               target_relative_distance[2 * i:2 * (i + 1)])
            current_distance = np.linalg.norm(current_relative_distance[2 * i:2 * (i + 1)] -
                                              target_relative_distance[2 * i:2 * (i + 1)])
            if current_distance < previous_distance:
                r = 1
                t = False
            else:
                r = -1
                t = False
            # 到达目标位置
            if abs(current_distance) <= 0.1:
                r = 5
                t = True
            reward.append(r)
            terminated.append(t)
        return reward, terminated

    def step(self, actions):
        self.step_ctr += 1
        # 移动前位置
        current_pos = self.pos.copy()

        # 一阶动力学
        for i in range(self._num_agent):
            self.vel[i] = self.action[actions[i]]
            self.pos[i] = self.pos[i] + self.vel[i]

        # leader 移动
        self.pos[-1] = self.pos[-1] + self.action[0] * 0.1

        # 只考虑移动后位置
        next_pos = self.pos.copy()
        # 计算奖励以及是否停止
        reward, terminated = self.calculate_reward(current_pos, next_pos)

        if self.is_render:
            self.render()

        # 超出时长
        if self.step_ctr > self.MAX_STEP:
            truncated = True
        else:
            truncated = False

        # 计算相对偏移
        current_pos = self.pos.flatten()
        next_state = np.dot(self.L, current_pos).reshape(self._num_agent, 2)

        # 超出通信范围
        for i in range(self._num_followers):
            if np.linalg.norm(next_state[i]) > 15:
                truncated = True
        return next_state, reward, truncated, terminated

    def render(self):
        plt.clf()

        for pos in self.pos:
            plt.scatter(pos[0], pos[1])
        plt.legend([1, 2, 3, 4, 0], loc='upper right')
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.pause(.001)


if __name__ == '__main__':
    # 编队队形
    formation = np.array([[0, 0],
                          [1, 0],
                          [0, 1]])
    # 网络拓扑
    # Adjacent matrix
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])

    env = Env(num_agent=3,
              formation=formation,
              adjacent_matrix=A,
              is_render=True,
              MAX_STEP=500)
