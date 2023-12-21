import numpy as np
import matplotlib.pyplot as plt
from blocks import Block


class Agent:
    def __init__(self, init_pos, init_vel, init_acc,
                 MAX_ACC=10, MAX_VEL=10, sampling_time=0.02):
        # 1x2 matrix
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
        # 避障安全系数
        self.k = 100

        self.history = [self.pos.copy()]

    def control(self, target, obstacles):
        self.acc = self.track(target=target) + self.obstacle_avoidance(obstacles)

    def obstacle_avoidance(self, obstacles):
        # 障碍物点云 obstacles Nx2
        # 过滤探测范围以外的点
        ind = np.logical_and(abs(obstacles[:, 0] - self.pos[0]) < self.detect_distance,
                             abs(obstacles[:, 1] - self.pos[1]) < self.detect_distance)
        if any(ind):
            # 如果存在障碍物
            obstacles = obstacles[ind]
            # 障碍物向量
            vector = (self.pos - obstacles)
            # 障碍物距离
            dis = (vector[:, 0] ** 2 + vector[:, 1] ** 2) ** 0.5
            # 障碍物方向
            weight = 1 / dis
            weight = weight.reshape(len(weight), 1)
            weight = np.hstack((weight, weight))
            direction = vector * weight
            acc1 = direction * weight * self.k
            acc1 = np.average(acc1, axis=0)
            # 扰动项，避免停止
            acc2 = np.random.rand(2) - 0.5
            return acc1 + acc2
        else:
            return 0

    def track(self, target):
        # 朝向目标移动
        acc1 = target - self.pos
        acc1 = np.clip(acc1, -self.MAX_ACC, self.MAX_ACC)
        # 稳定项
        acc2 = -2 * self.vel
        return acc1 + acc2

    def step(self, target, obstacles):
        self.control(target, obstacles)
        self.acc = np.clip(self.acc, -self.MAX_ACC, self.MAX_ACC)
        self.vel = self.vel + self.acc * self.sampling_time
        self.vel = np.clip(self.vel, -self.MAX_VEL, self.MAX_VEL)
        self.pos = self.pos + self.vel * self.sampling_time
        self.history.append(self.pos)


class Env:
    def __init__(self, num_leader,
                 num_follower,
                 formation_shape,
                 adjacent_matrix,
                 is_render=True,
                 low=0,
                 high=200,
                 MAX_STEP=500,
                 MAX_ACC=10,
                 MAX_VEL=10,
                 sampling_time=0.02):
        self.acc = None
        self.vel = None
        self.pos = None
        self._num_agent = num_leader + num_follower
        self._num_followers = num_follower
        self._num_leader = num_leader
        # 最大仿真时间
        self.MAX_STEP = MAX_STEP
        # 边界线
        self.low = low
        self.high = high
        # 重置环境
        self.reset()

        # 目标队形
        self.formation_shape = formation_shape
        # 邻接矩阵
        self.A = adjacent_matrix
        # 拉式矩阵
        self.L = self.A
        for i in range(self._num_agent):
            self.L[i, i] = -sum(self.L[i, :])
        # self.L = np.kron(self.L, np.eye(2))

        # 是否更新
        self.is_render = is_render
        # 仿真步数
        self.step_ctr = 0

        self.action = np.array([[0, 0],
                                [1, 0],
                                [0, 1],
                                [-1, 0],
                                [0, -1]]) * 0.1

        self.block = Block(radius_max=30, radius_min=20, low=self.low + 20, high=self.high - 20)

        self.obstacle = np.vstack(self.block.points)

        self.MAX_ACC = MAX_ACC
        self.MAX_VEL = MAX_VEL
        self.sampling_time = sampling_time
        # 生成智能体
        self.agent = [Agent(init_pos=self.pos[i],
                            init_vel=self.vel[i],
                            init_acc=self.acc[i],
                            MAX_ACC=self.MAX_ACC,
                            MAX_VEL=self.MAX_VEL,
                            sampling_time=self.sampling_time) for i in range(self._num_agent)]

    def reset(self):
        self.step_ctr = 0
        # 将机器人移动到随机位置
        # self.pos = np.round(np.random.rand(self._num_agent, 2) * 10 - 5, 1)
        self.pos = np.random.randint(self.low, self.low + 20, (self._num_agent, 2))
        self.acc = np.zeros(shape=(self._num_agent, 2))
        self.vel = np.zeros(shape=(self._num_agent, 2))

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

    def step(self, action=0):
        self.step_ctr += 1
        target = np.array([200, 200])
        # 障碍物点云
        obstacles = self.obstacle
        # 智能体移动
        for i in range(self._num_agent):
            self.agent[i].step(target=target, obstacles=obstacles)

        # 更新
        for i in range(self._num_agent):
            self.pos[i] = self.agent[i].pos

        # 计算奖励以及是否停止

        # 超出时长

        # 超出通信范围

        # 更新画面
        if self.is_render:
            self.render()

    def render(self):
        plt.cla()
        # 画障碍物
        # if self.step_ctr <= 1:
        for obs in self.block.points:
            plt.plot(obs[:, 0], obs[:, 1], c='b')
        # 画目的地
        plt.plot(200, 200, marker='*')

        # 画智能体
        for pos in self.pos:
            plt.plot(pos[0], pos[1], c='r', marker='o')

        for i in range(self._num_agent):
            his = np.vstack(self.agent[0].history)
            plt.plot(his[:, 0], his[:, 1])

        plt.xlim([-10, 200])
        plt.ylim([-10, 200])
        plt.axis('equal')
        plt.pause(.001)

    def show(self):
        pass


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

    env = Env(num_leader=1,
              num_follower=0,
              formation_shape=formation,
              adjacent_matrix=A,
              is_render=True,
              low=0,
              high=200,
              MAX_STEP=500,
              MAX_ACC=10,
              MAX_VEL=10,
              sampling_time=0.02)
    while True:
        env.step()

    #
    # # agent.control(target_pos=np.array([200, 200]), obstacles=circle.edge)
    # for i in range(1000):
    #     agent.control(target_pos=np.array([200, 200]), obstacles=circle.edge)
    #     agent.step(vel=0, acc=agent.acc)
    #     plt.cla()
    #     plt.xlim(-10, 250)
    #     plt.ylim(-10, 250)
    #     plt.plot(agent.pos[0], agent.pos[1], '*')
    #     plt.pause(.01)
    #     print(agent.pos[0], agent.pos[1])
