import numpy as np
import matplotlib.pyplot as plt
from blocks import Block


class Agent:
    def __init__(self, init_pos, init_vel, init_acc,
                 MAX_ACC=10, MAX_VEL=10, sampling_time=0.02):
        self.pos = init_pos
        self.vel = init_vel
        self.acc = init_acc

        # 最大加速度10cm
        self.MAX_ACC = MAX_ACC
        # 最大速递10cm
        self.MAX_VEL = MAX_VEL
        # 最小安全距离
        self.save_distance = self.MAX_VEL ** 2 / 2 / self.MAX_ACC
        # 探测距离=安全距离+5cm
        self.detect_distance = self.save_distance * 2
        self.sampling_time = sampling_time

        self.track = [self.pos]

    def control(self, target_pos, obstacles):
        # 朝向目标移动
        acc1 = target_pos - self.pos
        acc1 = np.clip(acc1, -self.MAX_ACC, self.MAX_ACC)
        # 稳定项
        acc2 = -2 * self.vel
        # 障碍物点云 obstacles Nx2
        ind = np.logical_and((obstacles[:, 0] - self.pos[0]) < self.detect_distance,
                             (obstacles[:, 1] - self.pos[1]) < self.detect_distance)
        obstacles = obstacles[ind]
        # 障碍物阻力
        acc3 = (self.pos - obstacles)
        dis = (acc3[:, 0] ** 2 + acc3[:, 1] ** 2) ** 0.5
        acc3 = np.average(acc3, axis=0)
        dis = np.linalg.norm(acc3)
        acc3 = acc3 / dis / (dis - 5)
        np.clip(acc3, -self.MAX_ACC, self.MAX_ACC)

        # 扰动项，避免停止
        acc4 = np.random.rand(2) - 0.5
        self.acc = acc1 + acc2 + acc3 + acc4

    def step(self, vel, acc=None):
        if acc is None:
            # 一阶运动学
            self.vel = np.clip(vel, -self.MAX_VEL, self.MAX_VEL)
            self.pos = self.pos + vel * self.sampling_time
        else:
            self.acc = np.clip(acc, -self.MAX_ACC, self.MAX_ACC)
            self.vel = self.vel + acc * self.sampling_time
            self.vel = np.clip(self.vel, -self.MAX_VEL, self.MAX_VEL)
            self.pos = self.pos + self.vel * self.sampling_time

        self.track.append(self.pos)


class Env:
    def __init__(self, num_leader,
                 num_follower,
                 formation_shape,
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
        self.formation_shape = formation_shape
        # 邻接矩阵
        self.A = adjacent_matrix
        # 拉式矩阵
        self.L = self.A
        for i in range(self._num_agent):
            self.L[i, i] = -sum(self.L[i, :])
        # self.L = np.kron(self.L, np.eye(2))

        self.is_render = is_render
        self.step_ctr = 0

        self._state_dim = self._num_agent
        self._action_dim = 5

        self.action = np.array([[0, 0],
                                [1, 0],
                                [0, 1],
                                [-1, 0],
                                [0, -1]]) * 0.1

        self.block = Block(radius_max=30, radius_min=20, low=20, high=200)

        self.track = []

    def state_space(self):
        return self._state_dim

    def action_space(self):
        return self._action_dim

    def reset(self):
        self.step_ctr = 0
        # 轨迹跟踪
        self.track = []
        # 将机器人移动到随机位置
        # self.pos = np.round(np.random.rand(self._num_agent, 2) * 10 - 5, 1)
        self.pos = np.array([[-3.8, -3.2],
                             [-2.9, -2.3],
                             [-1.9, -4.1],
                             [0.2, 4.1],
                             [0, 0]])
        self.acc = np.zeros(shape=(self._num_agent, 2))
        self.vel = np.zeros(shape=(self._num_agent, 2))

        # 计算相对偏移量
        current_pos = self.pos.flatten()
        rela_pos = np.dot(self.L, current_pos).reshape(self._num_agent, 2)
        current_vel = self.vel.flatten()
        rela_vel = np.dot(self.L, current_vel).reshape(self._num_agent, 2)
        state = np.hstack((rela_pos, rela_vel))
        self.track.append(self.pos)
        return state

    def action_sample(self):
        actions = np.random.rand(0, 5, self._num_agent)
        return actions

    # def calculate_reward(self, current_pos, next_pos):
    #     terminated = []
    #     reward = []
    #     # 以相对偏移方式计算
    #     # 考虑别的机器人的运动
    #     current_pos = current_pos.flatten()
    #     next_pos = next_pos.flatten()
    #     previous_relative_distance = np.dot(self.L, current_pos)
    #     current_relative_distance = np.dot(self.L, next_pos)
    #     target_relative_distance = np.dot(self.L, self.formation.flatten())
    #
    #     for i in range(self._num_agent):
    #         previous_distance = np.linalg.norm(previous_relative_distance[2 * i:2 * (i + 1)] -
    #                                            target_relative_distance[2 * i:2 * (i + 1)])
    #         current_distance = np.linalg.norm(current_relative_distance[2 * i:2 * (i + 1)] -
    #                                           target_relative_distance[2 * i:2 * (i + 1)])
    #         if current_distance < previous_distance:
    #             r = 1
    #             t = False
    #         else:
    #             r = -1
    #             t = False
    #         # 到达目标位置
    #         if abs(current_distance) <= 0.1:
    #             r = 5
    #             t = True
    #         reward.append(r)
    #         terminated.append(t)
    #     return reward, terminated

    def step(self, actions):
        self.step_ctr += 1
        # 移动前位置
        current_pos = self.pos.copy()

        # 一阶动力学
        for i in range(self._num_agent):
            self.vel[i] = self.action[actions[i]]

        # leader 移动
        self.vel[-1] = self.action[1] * 0.1

        # 更新
        self.pos = self.pos + self.vel

        # 只考虑移动后位置
        next_pos = self.pos.copy()
        # 计算奖励以及是否停止
        reward, terminated = self.calculate_reward(current_pos, next_pos)

        # 超出时长
        if self.step_ctr > self.MAX_STEP:
            truncated = True
        else:
            truncated = False

        current_pos = self.pos.flatten()
        rela_pos = np.dot(self.L, current_pos).reshape(self._num_agent, 2)
        current_vel = self.vel.flatten()
        rela_vel = np.dot(self.L, current_vel).reshape(self._num_agent, 2)
        next_state = np.hstack((rela_pos, rela_vel))
        self.track.append(self.pos)

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

    def show(self):
        track = np.hstack(self.track)

        for t in self.track:
            for pos in t:
                plt.scatter(pos[0], pos[1])
            plt.legend([1, 2, 3, 4, 0], loc='upper right')
            plt.xlim([-20, 20])
            plt.ylim([-20, 20])
            plt.pause(.01)
            plt.cla()
        return


if __name__ == '__main__':
    # 编队队形
    # formation = np.array([[0, 0],
    #                       [1, 0],
    #                       [0, 1]])
    # # 网络拓扑
    # # Adjacent matrix
    # A = np.array([[0, 1, 0],
    #               [0, 0, 1],
    #               [1, 0, 0]])
    #
    # env = Env(num_leader=1,
    #           num_follower=0,
    #           formation_shape=formation,
    #           adjacent_matrix=A,
    #           is_render=True,
    #           MAX_STEP=500)

    class Circle:
        def __init__(self, center, radius_max, radius_min, ):
            r = radius_min + (radius_max - radius_min) * np.random.rand()

            num_nodes = round(2 * np.pi * r / 0.1)
            self.edge = []
            for i in range(num_nodes + 1):
                p = np.array([[center[0] + r * np.cos(i / num_nodes * 2 * np.pi),
                               center[1] + r * np.sin(i / num_nodes * 2 * np.pi)]])
                self.edge.append(p)

            self.edge = np.vstack(self.edge)


    circle = Circle(center=np.array([10, 10]), radius_max=10, radius_min=5)

    agent = Agent(init_pos=np.array([0, 0]),
                  init_vel=np.array([0, 0]),
                  init_acc=np.array([0, 0]), sampling_time=0.1)

    # agent.control(target_pos=np.array([200, 200]), obstacles=circle.edge)
    for i in range(1000):
        agent.control(target_pos=np.array([200, 200]), obstacles=circle.edge)
        agent.step(vel=0, acc=agent.acc)
        plt.cla()
        plt.xlim(-10, 250)
        plt.ylim(-10, 250)
        plt.plot(agent.pos[0], agent.pos[1], '*')
        plt.pause(.01)
        print(agent.pos[0], agent.pos[1])
