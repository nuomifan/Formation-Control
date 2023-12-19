import numpy as np


class Agent:
    def __init__(self, id, L, formation, acc=0.01):
        self.id = id
        self.L = L
        self.formation = formation

        self.pos = np.array([0, 0])
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])

        self.action_list = np.array([[1, 0],
                                     [0, 1],
                                     [-1, 0],
                                     [0, -1],
                                     [0, 0]]) * acc

        self.L[id] = -sum(self.L)
        self.L = np.kron(self.L, np.eye(2))

        self.MAX_ACC = 4
        self.MAX_VEL = 4
        self.MAX_XY = 10
        self.previous_distance = None

    def step(self, action):
        self.acc = self.acc + self.action_list[action]
        self.acc = np.clip(self.acc, -self.MAX_ACC, self.MAX_ACC)
        self.vel = self.vel + self.acc
        self.vel = np.clip(self.vel, -self.MAX_VEL, self.MAX_VEL)
        self.pos = self.pos + self.vel
        self.pos = np.clip(self.pos, -self.MAX_XY, self.MAX_XY)
        return self.acc, self.vel, self.pos

    def reset(self):
        self.pos = np.random.randint(-5, 5, 2)
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])
        self.previous_distance = None
        return self.acc, self.vel, self.pos

    def calculate_reward(self, all_pos):
        # relative distance
        current = np.inner(self.L, all_pos)
        target = np.inner(self.L, self.formation.flatten())
        distance = np.linalg.norm(current - target)
        if self.previous_distance is None:
            self.previous_distance = distance
        reward = self.previous_distance - distance
        self.previous_distance = distance
        return reward
