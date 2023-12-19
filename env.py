import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

agent_pos = np.array([8, 1, 5, 5, 6, 2])


class env():
    def __init__(self):
        # formation target
        self.formation_target = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]) * 5

        self.reset()

    def reset(self):

        # random initial position 记录初始位置，为调用过去的控制算法保留初值
        self.pos_ini = np.random.random((8, 2)) * 10
        # self.pos_ini = np.array([[1.,1],[8,8],[0,5],[3,4],[7,8],[9,8],[1,7],[4,9]])
        self.pos = self.pos_ini.copy()
        self.old_pos = self.pos.copy()

        # 初始速度为0，初始加速度也为0
        self.velocity = np.zeros([8, 2])
        self.accelerate = np.zeros([8, 2])
        # energy function
        self.E = self.energy()

        # agent_track
        self.agent_track = np.array([self.pos])

        # agent energy track
        self.agent_energy_track = self.E.copy()

        return self.pos.reshape(16).copy()

    def reward(self):
        reward = np.array([])
        nE = self.energy()
        oE = self.E
        for i in range(8):
            if nE[i] > oE[i]:
                reward = np.append(reward, -1)
            elif nE[i] == oE[i]:
                reward = np.append(reward, 0)
            else:
                reward = np.append(reward, 1)
        self.E = nE
        return reward

    def step(self, action):
        self.dynamic(action)
        reward = self.reward()
        self.old_pos = self.pos.copy()

        # record information
        dim = self.agent_track.shape
        self.agent_track = np.append(self.agent_track, self.pos).reshape(dim[0] + 1, dim[1], dim[2])
        self.agent_energy_track = np.c_[self.agent_energy_track, self.E]

        if np.linalg.norm(self.E) < 1:
            done = True
        else:
            done = False
        return self.pos.reshape(16).copy(), reward, done

    def plt(self, flag):
        clc = ['k', 'r', 'y', 'g', 'b', 'c', 'm', 'tan']
        if flag == 'start':
            # figure handle
            self.fig = plt.figure()
            self.subfig1 = self.fig.add_subplot(221)
            self.subfig2 = self.fig.add_subplot(222)
            self.subfig3 = self.fig.add_subplot(223)
            self.subfig4 = self.fig.add_subplot(224)
            plt.ion()
            formation_target_x = self.formation_target[:, 0]
            formation_target_y = self.formation_target[:, 1]
            x = self.pos[:, 0]
            y = self.pos[:, 1]
            self.subfig1.scatter(formation_target_x, formation_target_y)
            self.subfig1.scatter(x, y)
            plt.subplot(221)
            for i in range(8):
                plt.text(x[i], y[i], str(i))
                plt.text(formation_target_x[i], formation_target_y[i], 'refer' + str(i))
        elif flag == 'update':
            x = self.pos[:, 0]
            y = self.pos[:, 1]
            self.subfig2.clear()
            self.subfig4.clear()
            self.subfig2.scatter(x, y)
            plt.subplot(222)
            for i in range(8):
                plt.text(x[i], y[i], str(i))
            x = self.agent_track[:, :, 0]
            y = self.agent_track[:, :, 1]
            self.subfig4.plot(x, y)
            plt.pause(.0001)
        elif flag == 'finish':
            for i in range(8):
                ey = self.agent_energy_track[i, :]
                self.subfig3.plot(ey)
            plt.pause(5)
        elif flag == 'clean':
            plt.close('all')

    def energy(self):
        # new Energy consider other agent action
        # E0 = np.linalg.norm(self.pos[0] - self.pos[1] - (self.formation_target[0] - self.formation_target[1]))
        # E1 = np.linalg.norm(self.pos[1] - self.pos[2] - (self.formation_target[1] - self.formation_target[2]))
        # E2 = np.linalg.norm(self.pos[2] - self.pos[3] - (self.formation_target[2] - self.formation_target[3]))
        # E3 = np.linalg.norm(self.pos[3] - self.pos[4] - (self.formation_target[3] - self.formation_target[4]))
        # E4 = np.linalg.norm(self.pos[4] - self.pos[5] - (self.formation_target[4] - self.formation_target[5]))
        # E5 = np.linalg.norm(self.pos[5] - self.pos[6] - (self.formation_target[5] - self.formation_target[6]))
        # E6 = np.linalg.norm(self.pos[6] - self.pos[7] - (self.formation_target[6] - self.formation_target[7]))
        # E7 = np.linalg.norm(self.pos[7] - self.pos[0] - (self.formation_target[7] - self.formation_target[0]))

        # not consider other agent action
        E0 = np.linalg.norm(self.pos[0] - self.old_pos[1] - (self.formation_target[0] - self.formation_target[1]))
        E1 = np.linalg.norm(self.pos[1] - self.old_pos[2] - (self.formation_target[1] - self.formation_target[2]))
        E2 = np.linalg.norm(self.pos[2] - self.old_pos[3] - (self.formation_target[2] - self.formation_target[3]))
        E3 = np.linalg.norm(self.pos[3] - self.old_pos[4] - (self.formation_target[3] - self.formation_target[4]))
        E4 = np.linalg.norm(self.pos[4] - self.old_pos[5] - (self.formation_target[4] - self.formation_target[5]))
        E5 = np.linalg.norm(self.pos[5] - self.old_pos[6] - (self.formation_target[5] - self.formation_target[6]))
        E6 = np.linalg.norm(self.pos[6] - self.old_pos[7] - (self.formation_target[6] - self.formation_target[7]))
        E7 = np.linalg.norm(self.pos[7] - self.old_pos[0] - (self.formation_target[7] - self.formation_target[0]))

        nE = np.array([E0, E1, E2, E3, E4, E5, E6, E7])
        return nE

    def traditional_position_based(self):
        x = self.pos_ini
        x = x.reshape(16)
        y = np.zeros([16, 1000])
        # 采样时间
        e = 0.01
        # P控制
        kp = 2

        A = np.eye(16) * (1 - e * kp)
        B = e * kp * self.formation_target.reshape(16)
        for i in range(1000):
            y[:, i] = x
            x = np.dot(A, x) + B
        for i in range(8):
            plt.plot(y[2 * i, :], y[2 * i + 1, :])

        formation_target_x = self.formation_target[:, 0]
        formation_target_y = self.formation_target[:, 1]
        self.subfig2.scatter(formation_target_x, formation_target_y)
        plt.show()

    def traditional_relatvie_position_based(self):

        total_step = 200
        x = self.pos_ini
        x = x.reshape(16)
        y = np.zeros([16, total_step])

        # 采样时间
        e = 0.01
        # P控制
        kp = 2

        # 拉氏矩阵
        L = np.zeros([8, 8])
        for i in range(7):
            L[i, i + 1] = 1
            L[i, i] = -1
        L[7, 0] = 1
        L[7, 7] = -1

        L = np.kron(L, np.eye(2))

        for i in range(total_step):
            # 误差
            ep = self.formation_target.reshape(16) - x
            y[:, i] = x
            x = e * kp * np.dot((np.eye(16) - L), ep) + x

        for i in range(8):
            plt.plot(y[2 * i, :], y[2 * i + 1, :])
        formation_target_x = self.formation_target[:, 0]
        formation_target_y = self.formation_target[:, 1]
        self.subfig2.scatter(formation_target_x, formation_target_y)
        plt.show()

        print(1)

    def dynamic(self, action, order=1):
        action_list = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]) * 0.1
        if order == 1:
            # 一阶模型
            for i in range(8):
                self.velocity[i] = action_list[action[i]]
                self.pos[i] = self.pos[i] + self.velocity[i]
                if any(self.pos[i] > 100) or any(self.pos[i] < 0):
                    self.pos[i] = self.old_pos[i]
        else:
            # 二阶模型
            for i in range(8):
                self.accelerate[i] = action_list[action[i]]
                self.velocity[i] = self.velocity[i] + self.accelerate[i]
                self.pos[i] = self.pos[i] + self.velocity[i]
                if any(self.pos[i] > 100) or any(self.pos[i] < 0):
                    self.pos[i] = self.old_pos[i]
