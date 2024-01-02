import numpy as np


class Agent:
    def __init__(self, init_pos, init_vel, init_acc, id,
                 MAX_ACC=10, MAX_VEL=10, sampling_time=0.02, leader=True):
        self._leader = leader
        # 机器人编号
        self._id = id

        # 1x2 matrix
        self._pos = init_pos
        self._vel = init_vel
        self._acc = init_acc

        # 最大加速度10cm
        self._MAX_ACC = MAX_ACC
        # 最大速递10cm
        self._MAX_VEL = MAX_VEL
        # 最小安全距离
        self._save_distance = self._MAX_VEL ** 2 / 2 / self._MAX_ACC
        # 探测距离=安全距离+5cm
        self._detect_distance = self._save_distance * 2
        self._sampling_time = sampling_time
        # 避障安全系数
        self._k = self._save_distance * self._MAX_VEL
        self._history = []

    def history(self):
        return self._history

    def state(self):
        return self._pos, self._vel, self._acc

    def reset(self, pos, vel, acc):
        self._pos = pos
        self._vel = vel
        self._acc = acc
        self._history = []

    def control(self, target, obstacles, all_agents):
        # 带衰减的轨迹跟踪控制信号
        self._acc = self.track(target=target)
        # 传统的避障控制信号
        other_agents = np.delete(all_agents, self._No, axis=0)
        obstacles = np.vstack((obstacles, other_agents))
        self._acc += self.obstacle_avoidance(obstacles)

    def obstacle_avoidance(self, obstacles):
        # 障碍物点云 obstacles Nx2
        # 过滤探测范围以外的点
        ind = np.logical_and(abs(obstacles[:, 0] - self._pos[0]) < self._detect_distance,
                             abs(obstacles[:, 1] - self._pos[1]) < self._detect_distance)
        if any(ind):
            # 如果存在障碍物
            obstacles = obstacles[ind]
            # 障碍物向量
            vector = (self._pos - obstacles)
            # 障碍物距离
            dis = (vector[:, 0] ** 2 + vector[:, 1] ** 2) ** 0.5
            # 障碍物方向
            weight = 1 / dis
            weight = weight.reshape(len(weight), 1)
            weight = np.hstack((weight, weight))
            direction = vector * weight
            acc1 = direction * weight
            acc1 = np.average(acc1, axis=0)
            acc1 = acc1 / np.linalg.norm(acc1)
            if dis.min() < 10:
                print(dis.min())
            acc1 = acc1 * 1 / dis.min() * self._k
            # 扰动项，避免停止
            acc2 = np.random.rand(2) - 0.5
            return acc1 + acc2
        else:
            return 0

    def track(self, target):
        # 朝向目标移动
        acc1 = target - self._pos
        acc1 = np.clip(acc1, -self._MAX_ACC, self._MAX_ACC)
        # 稳定项
        acc2 = -2 * self._vel
        return acc1 + acc2

    def step(self, target, obstacles, all_agents):

        # 轨迹跟踪+避障，计算加速度_acc
        self.control(target, obstacles, all_agents)
        # 质点二阶动力学更新
        self._acc = np.clip(self._acc, -self._MAX_ACC, self._MAX_ACC)
        self._vel = self._vel + self._acc * self._sampling_time
        self._vel = np.clip(self._vel, -self._MAX_VEL, self._MAX_VEL)
        self._pos = self._pos + self._vel * self._sampling_time
        # 记录轨迹
        self._history.append(self._pos)

    @property
    def pos(self):
        return self._pos

