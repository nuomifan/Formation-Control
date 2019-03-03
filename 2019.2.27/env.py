import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

agent_pos = np.array([8, 1, 5, 5, 6, 2])


class env():
    def __init__(self):
        # random initial position
        self.pos = np.random.random((8, 2)) * 10
        self.old_pos = self.pos.copy()
        # formation target
        self.ft = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]) * 5
        # energy function
        E0 = np.linalg.norm(self.pos[0] - self.pos[1] - (self.ft[0] - self.ft[1]))
        E1 = np.linalg.norm(self.pos[1] - self.pos[2] - (self.ft[1] - self.ft[2]))
        E2 = np.linalg.norm(self.pos[2] - self.pos[3] - (self.ft[2] - self.ft[3]))
        E3 = np.linalg.norm(self.pos[3] - self.pos[4] - (self.ft[3] - self.ft[4]))
        E4 = np.linalg.norm(self.pos[4] - self.pos[5] - (self.ft[4] - self.ft[5]))
        E5 = np.linalg.norm(self.pos[5] - self.pos[6] - (self.ft[5] - self.ft[6]))
        E6 = np.linalg.norm(self.pos[6] - self.pos[7] - (self.ft[6] - self.ft[7]))
        E7 = np.linalg.norm(self.pos[7] - self.pos[0] - (self.ft[7] - self.ft[0]))
        self.E = np.array([E0, E1, E2, E3, E4, E5, E6, E7])
        # agent_track 8*2*n array
        # 引用第i时刻第j个智能体的k坐标
        # self.atra[i][j][k]
        self.atra = np.array([self.pos])
        # agent energy track
        self.aetra = self.E.copy()

    def reset(self):
        # random initial position
        self.pos = np.random.random((8, 2)) * 10
        self.old_pos = self.pos.copy()
        # formation target
        self.ft = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]) * 5
        # energy function
        E0 = np.linalg.norm(self.pos[0] - self.pos[1] - (self.ft[0] - self.ft[1]))
        E1 = np.linalg.norm(self.pos[1] - self.pos[2] - (self.ft[1] - self.ft[2]))
        E2 = np.linalg.norm(self.pos[2] - self.pos[3] - (self.ft[2] - self.ft[3]))
        E3 = np.linalg.norm(self.pos[3] - self.pos[4] - (self.ft[3] - self.ft[4]))
        E4 = np.linalg.norm(self.pos[4] - self.pos[5] - (self.ft[4] - self.ft[5]))
        E5 = np.linalg.norm(self.pos[5] - self.pos[6] - (self.ft[5] - self.ft[6]))
        E6 = np.linalg.norm(self.pos[6] - self.pos[7] - (self.ft[6] - self.ft[7]))
        E7 = np.linalg.norm(self.pos[7] - self.pos[0] - (self.ft[7] - self.ft[0]))
        self.E = np.array([E0, E1, E2, E3, E4, E5, E6, E7])
        # agent_track 8*2*n array
        # 引用第i时刻第i个智能体的x，y坐标
        # self.atra[i][j][k]
        self.atra = np.array([self.pos])
        # agent energy track
        self.aetra = self.E.copy()
        return self.pos

    def reward(self):
        reward = np.array([])
        # new Energy
        E0 = np.linalg.norm(self.pos[0] - self.old_pos[1] - (self.ft[0] - self.ft[1]))
        E1 = np.linalg.norm(self.pos[1] - self.old_pos[2] - (self.ft[1] - self.ft[2]))
        E2 = np.linalg.norm(self.pos[2] - self.old_pos[3] - (self.ft[2] - self.ft[3]))
        E3 = np.linalg.norm(self.pos[3] - self.old_pos[4] - (self.ft[3] - self.ft[4]))
        E4 = np.linalg.norm(self.pos[4] - self.old_pos[5] - (self.ft[4] - self.ft[5]))
        E5 = np.linalg.norm(self.pos[5] - self.old_pos[6] - (self.ft[5] - self.ft[6]))
        E6 = np.linalg.norm(self.pos[6] - self.old_pos[7] - (self.ft[6] - self.ft[7]))
        E7 = np.linalg.norm(self.pos[7] - self.old_pos[0] - (self.ft[7] - self.ft[0]))
        nE = np.array([E0, E1, E2, E3, E4, E5, E6, E7])
        oE = self.E
        for i in range(8):
            if nE[i] > oE[i]:
                reward = np.append(reward, -1)
            else:
                reward = np.append(reward, 1)
        self.E = nE
        return reward

    def step(self, actionlist):
        for i in range(7):
            self.pos[i] = self.pos[i] + actionlist[i]
        reward = self.reward()
        self.old_pos = self.pos.copy()
        dim = self.atra.shape
        self.atra = np.append(self.atra, self.pos).reshape(dim[0] + 1, dim[1], dim[2])
        self.aetra = np.c_[self.aetra, self.E]
        if np.linalg.norm(self.E) < 1:
            done = True
        else:
            done = False
        return self.pos, reward, done

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
            ftx = self.ft[:, 0]
            fty = self.ft[:, 1]
            x = self.pos[:, 0]
            y = self.pos[:, 1]
            self.subfig1.scatter(ftx, fty)
            self.subfig1.scatter(x, y)
        elif flag == 'update':
            x = self.pos[:, 0]
            y = self.pos[:, 1]
            self.subfig2.clear()
            self.subfig4.clear()
            self.subfig2.scatter(x, y)
            x = self.atra[:, :, 0]
            y = self.atra[:, :, 1]
            self.subfig4.plot(x, y)
            plt.pause(.001)
        elif flag == 'finish':
            for i in range(8):
                ey = self.aetra[i, :]
                self.subfig3.plot(ey)
            plt.pause(5)
        elif flag == 'clean':
            plt.close('all')
