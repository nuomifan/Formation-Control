import numpy as np
import matplotlib.pyplot as plt
import os
from blocks import Block
from agent import Agent
from util import del_file


class Env:
    def __init__(self, num_leader,
                 num_follower,
                 formation_shape,
                 adjacent_matrix,
                 is_render=True,
                 low=0,
                 high=200,
                 MAX_STEP=5000,
                 MAX_ACC=10,
                 MAX_VEL=10,
                 sampling_time=0.02):
        self._num_agent = num_leader + num_follower
        self._num_followers = num_follower
        self._num_leader = num_leader
        # 最大仿真时间
        self.MAX_STEP = MAX_STEP
        # 边界线
        self.low = low
        self.high = high

        # 机器人移动从(-50,-50)到(200,200)
        # 将机器人移动到随机位置
        self.pos = np.random.uniform(self.low - 50, self.low, (self._num_agent, 2))
        self.acc = np.zeros(shape=(self._num_agent, 2))
        self.vel = np.zeros(shape=(self._num_agent, 2))
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

        self.block = Block(radius_max=30, radius_min=20, low=self.low + 20, high=self.high - 20)

        self.obstacle = np.vstack(self.block.points)

        self.MAX_ACC = MAX_ACC
        self.MAX_VEL = MAX_VEL
        self.sampling_time = sampling_time
        # 生成智能体
        self.leader_agent = [Agent(init_pos=self.pos[i],
                                   init_vel=self.vel[i],
                                   init_acc=self.acc[i],
                                   id=i,
                                   Laplacian=self.L,
                                   MAX_ACC=self.MAX_ACC,
                                   MAX_VEL=self.MAX_VEL,
                                   sampling_time=self.sampling_time) for i in range(self._num_leader)]
        self.follower_agent = [Agent(init_pos=self.pos[i],
                                     init_vel=self.vel[i],
                                     init_acc=self.acc[i],
                                     id=i,
                                     Laplacian=self.L,
                                     MAX_ACC=self.MAX_ACC,
                                     MAX_VEL=self.MAX_VEL,
                                     sampling_time=self.sampling_time) for i in range(self._num_followers)]
        self.fig, self.ax = plt.subplots()

    def load_block_map(self, dir='block map/1/'):
        self.block.recover_block_map(dir)
        self.obstacle = np.vstack(self.block.points)

    def reset(self):
        self.step_ctr = 0
        # 将机器人移动到随机位置
        # 机器人移动从(-50,-50)到(200,200)
        # 将机器人移动到随机位置
        self.pos = np.random.uniform(self.low - 50, self.low, (self._num_agent, 2))
        self.acc = np.zeros(shape=(self._num_agent, 2))
        self.vel = np.zeros(shape=(self._num_agent, 2))

        for i in range(self._num_agent):
            self.agent[i].reset(self.pos[i], self.vel[i], self.acc[i])

    def step(self, action=0):
        self.step_ctr += 1
        target = np.array([250, 250])
        # 障碍物点云
        obstacles = self.obstacle
        # 智能体移动
        for i in range(self._num_agent):
            self.agent[i].step(target=target, obstacles=obstacles, all_agents=self.pos)

        # 更新
        for i in range(self._num_agent):
            self.pos[i], self.vel[i], self.acc[i] = self.agent[i].state()

        # 超出时长
        if self.step_ctr >= self.MAX_STEP:
            return True

        # 超出通信范围

        # 更新画面
        if self.is_render:
            self.render()

        # 计算奖励以及是否停止
        if np.linalg.norm(target - self.agent[0].pos) < 0.1:
            return True
        else:
            return False

    def render(self):
        # if self.step_ctr <= 1:
        # self.fig, self.ax = plt.subplots()
        # 画障碍物
        for b in self.block.block:
            if b["shape"] == 'circle':
                circle = plt.Circle((b['vertex'][0], b['vertex'][1]), b['vertex'][2], fill=True, color="blue")
                self.ax.add_artist(circle)
            if b["shape"] == 'rectangle':
                self.ax.fill(b['vertex'][:, 0], b['vertex'][:, 1], fill=True, color='blue')
            if b["shape"] == 'triangle':
                self.ax.fill(b['vertex'][:, 0], b['vertex'][:, 1], fill=True, color='blue')

        # 画目的地
        self.ax.plot(250, 250, marker='*')

        # 画智能体
        for pos in self.pos:
            self.ax.plot(pos[0], pos[1], marker='o')

        for i in range(self._num_agent):
            his = np.vstack(self.agent[i].history())
            self.ax.plot(his[:, 0], his[:, 1], color='red')

        self.ax.text(-30, 250, "velocity: [%d, %d]" % (self.vel[0, 0], self.vel[0, 1]), fontsize=6)
        self.ax.text(-30, 240, "position: [%d, %d]" % (self.pos[0, 0], self.pos[0, 1]), fontsize=6)

        self.ax.set_xlim(-60, 300)
        self.ax.set_ylim(-60, 300)

        plt.pause(.1)
        self.ax.cla()

    def store_transition(self):
        if not os.path.exists("history"):
            os.mkdir("history")
        fold_num = len(os.listdir("history"))
        path = 'history/%d/' % (fold_num + 1)
        os.mkdir(path)
        # 画障碍物
        for b in self.block.block:
            if b["shape"] == 'circle':
                circle = plt.Circle((b['vertex'][0], b['vertex'][1]), b['vertex'][2], fill=True, color="blue")
                self.ax.add_artist(circle)
            if b["shape"] == 'rectangle':
                self.ax.fill(b['vertex'][:, 0], b['vertex'][:, 1], fill=True, color='blue')
            if b["shape"] == 'triangle':
                self.ax.fill(b['vertex'][:, 0], b['vertex'][:, 1], fill=True, color='blue')

        # 画目的地
        self.ax.plot(250, 250, marker='*')

        for i in range(self._num_agent):
            his = np.vstack(self.agent[i].history())
            # 画起始点
            self.ax.plot(his[0, 0], his[0, 1], marker='o')
            # 画轨迹
            self.ax.plot(his[:, 0], his[:, 1], color='red')

        self.ax.set_xlim(-60, 300)
        self.ax.set_ylim(-60, 300)
        plt.savefig(path + 'track.png')
        self.ax.cla()


if __name__ == '__main__':
    del_file('history')
    # 编队队形
    formation = np.array([[0, 0],
                          [10, 0],
                          [0, 10]])
    # 网络拓扑
    # Adjacent matrix
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])

    env = Env(num_leader=1,
              num_follower=2,
              formation_shape=formation,
              adjacent_matrix=A,
              is_render=False,
              low=0,
              high=200,
              MAX_STEP=500,
              MAX_ACC=20,
              MAX_VEL=20,
              sampling_time=0.02)
    for i in range(20):
        env.reset()
        env.load_block_map('block map/%d/' % (i + 1))
        while True:
            terminated = env.step()
            if terminated:
                env.store_transition()
                break
