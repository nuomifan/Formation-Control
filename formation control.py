import matplotlib.pyplot as plt
import numpy as np


class Formation:
    def __init__(self, num_leader,
                 num_follower,
                 formation,
                 adjacent_matrix,
                 init_state,
                 sample_time=0.02):
        self._num_agent = num_leader + num_follower
        self._num_followers = num_follower
        self._num_leader = num_leader
        # 目标队形
        self.formation = formation
        # 邻接矩阵
        self.A = adjacent_matrix
        # 拉式矩阵
        self.L = self.A
        for i in range(self._num_agent):
            self.L[i, i] = -sum(self.L[i, :])
        # self.L = np.kron(self.L, np.eye(2))

        self.sample_time = sample_time
        self.track = []
        self.init_state = init_state

    def formation_control(self):
        # x_dot = -Lx
        # x(k+1)-x(k) = -eLx(k)
        # x(k+1) = (I-eL)x(k)
        I = np.eye(len(self.L))
        A = I + self.sample_time * self.L
        x = self.init_state
        for i in range(1000):
            x = np.dot(A, x) - formation_shape * self.sample_time
            self.track.append(x)
        t = np.hstack(self.track)
        for i in range(self._num_agent):
            plt.plot(t[i, :], label=str(i))
        plt.legend()
        plt.pause(.1)
        print(t)


if __name__ == '__main__':
    # 网络拓扑
    # Adjacent matrix
    A = np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])

    x0 = np.array([1, 2, 3, 4, 5]).reshape(5, 1)

    formation_shape = np.array([3, 2, 3, 1, 0.1]).reshape(5, 1)

    f = Formation(num_leader=1,
                  num_follower=4,
                  formation=formation_shape,
                  adjacent_matrix=A,
                  init_state=x0,
                  sample_time=0.02)

    f.formation_control()
