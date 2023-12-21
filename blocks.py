import numpy as np
import matplotlib.pyplot as plt
import os
from util import del_file


class Triangle:
    def __init__(self, center, radius_max, radius_min):
        r1 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta1 = np.random.rand() * 2 / 3 * np.pi
        r2 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta2 = np.random.rand() * 2 / 3 * np.pi + 2 / 3 * np.pi
        r3 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta3 = np.random.rand() * 2 / 3 * np.pi + 4 / 3 * np.pi
        self.vertex = np.array([[center[0] + r1 * np.cos(theta1), center[1] + r1 * np.sin(theta1)],
                                [center[0] + r2 * np.cos(theta2), center[1] + r2 * np.sin(theta2)],
                                [center[0] + r3 * np.cos(theta3), center[1] + r3 * np.sin(theta3)]])

        e1 = (self.vertex[1] - self.vertex[0]) / np.linalg.norm(self.vertex[0] - self.vertex[1]) * 0.1

        e2 = (self.vertex[2] - self.vertex[1]) / np.linalg.norm(self.vertex[1] - self.vertex[2]) * 0.1

        e3 = (self.vertex[0] - self.vertex[2]) / np.linalg.norm(self.vertex[2] - self.vertex[0]) * 0.1

        self.edge = [self.vertex[0]]

        while True:
            p = self.edge[-1] + e1
            if np.linalg.norm(p - self.vertex[1]) < 0.1:
                break
            self.edge.append(p)

        self.edge.append(self.vertex[1])

        while True:
            p = self.edge[-1] + e2
            if np.linalg.norm(p - self.vertex[2]) < 0.1:
                break
            self.edge.append(p)

        self.edge.append(self.vertex[2])

        while True:
            p = self.edge[-1] + e3
            if np.linalg.norm(p - self.vertex[0]) < 0.1:
                break
            self.edge.append(p)

        self.edge.append(self.vertex[0])

        self.edge = np.vstack(self.edge)


class Rectangle:
    def __init__(self, center, radius_max, radius_min, ):
        r = radius_min + (radius_max - radius_min) * np.random.rand()
        theta1 = np.random.rand() * 2 * np.pi
        theta2 = theta1 + np.random.rand() * np.pi / 2 + np.pi / 4
        theta3 = theta1 + np.pi
        theta4 = theta2 + np.pi
        self.vertex = np.array([[center[0] + r * np.cos(theta1), center[1] + r * np.sin(theta1)],
                                [center[0] + r * np.cos(theta2), center[1] + r * np.sin(theta2)],
                                [center[0] + r * np.cos(theta3), center[1] + r * np.sin(theta3)],
                                [center[0] + r * np.cos(theta4), center[1] + r * np.sin(theta4)]])

        e1 = (self.vertex[1] - self.vertex[0]) / np.linalg.norm(self.vertex[0] - self.vertex[1]) * 0.1

        e2 = (self.vertex[2] - self.vertex[1]) / np.linalg.norm(self.vertex[1] - self.vertex[2]) * 0.1

        e3 = (self.vertex[3] - self.vertex[2]) / np.linalg.norm(self.vertex[3] - self.vertex[2]) * 0.1

        e4 = (self.vertex[0] - self.vertex[3]) / np.linalg.norm(self.vertex[3] - self.vertex[0]) * 0.1

        self.edge = [self.vertex[0]]

        while True:
            p = self.edge[-1] + e1
            if np.linalg.norm(p - self.vertex[1]) < 0.1:
                break
            self.edge.append(p)

        self.edge.append(self.vertex[1])

        while True:
            p = self.edge[-1] + e2
            if np.linalg.norm(p - self.vertex[2]) < 0.1:
                break
            self.edge.append(p)

        self.edge.append(self.vertex[2])

        while True:
            p = self.edge[-1] + e3
            if np.linalg.norm(p - self.vertex[3]) < 0.1:
                break
            self.edge.append(p)

        self.edge.append(self.vertex[3])

        while True:
            p = self.edge[-1] + e4
            if np.linalg.norm(p - self.vertex[0]) < 0.1:
                break
            self.edge.append(p)

        self.edge.append(self.vertex[0])

        self.edge = np.vstack(self.edge)


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


class Block:
    def __init__(self, radius_max=30, radius_min=20, low=20, high=200):
        self.points = []
        self.blocks_center = []
        self.radius_max = radius_max
        self.radius_min = radius_min
        self.low = low
        self.high = high
        self.generate_centers()
        self.generate_blocks()

    def generate_centers(self):
        self.points = []
        self.blocks_center = []
        # 生成障碍物中心，假设车的大小是10cm，障碍物直径30-50cm左右，
        # 假设实验场地是2X2米的范围，那么就是200cm x 200cm，
        # 尝试10000次随机中心，生成尽可能多的障碍物
        center = np.random.randint(self.low, self.high, (1, 2))
        self.blocks_center.append(center)
        for i in range(10000):
            new_center = np.random.randint(self.low, self.high, (1, 2))
            centers = np.vstack(self.blocks_center)
            dis = new_center - centers
            if all((dis[:, 0] ** 2 + dis[:, 1] ** 2) ** 0.5 > self.radius_max * 2):
                self.blocks_center.append(new_center)

        self.blocks_center = np.vstack(self.blocks_center)

    def generate_blocks(self):
        for i in range(len(self.blocks_center)):
            shape = np.random.choice(['triangle', 'rectangle', 'circle'])
            if shape == 'triangle':
                self.points.append(Triangle(center=self.blocks_center[i],
                                            radius_max=self.radius_max,
                                            radius_min=self.radius_min).edge)
            if shape == 'rectangle':
                self.points.append(Rectangle(center=self.blocks_center[i],
                                             radius_max=self.radius_max,
                                             radius_min=self.radius_min).edge)
            if shape == 'circle':
                self.points.append(Circle(center=self.blocks_center[i],
                                          radius_max=self.radius_max,
                                          radius_min=self.radius_min).edge)

    def save_block_map(self):
        fold_num = len(os.listdir("block map"))
        path = 'block map/%d/' % (fold_num + 1)
        os.mkdir(path)
        np.savetxt(path + 'block center.txt', self.blocks_center)
        for i in range(len(self.blocks_center)):
            np.savetxt(path + 'block_%d.txt' % i, self.points[i])
        plt.clf()
        for obs in self.points:
            plt.plot(obs[:, 0], obs[:, 1], c='b')
        plt.savefig(path + '{}.png'.format(fold_num + 1))

    def plot(self):
        plt.clf()
        for obs in self.points:
            plt.plot(obs[:, 0], obs[:, 1], c='b')
        plt.pause(.1)

    def recover_block_map(self, dir='block map/1/'):
        self.blocks_center = []
        self.points = []
        self.blocks_center = np.loadtxt(dir + 'block center.txt')
        for i in range(len(self.blocks_center)):
            self.points.append(np.loadtxt(dir + 'block_%d.txt'%i))


if __name__ == '__main__':
    # del_file('block map')
    b = Block(radius_max=30, radius_min=20, low=20, high=200)
    b.recover_block_map()
    b.plot()
    print(1)
    # b.save_block_map()
    # for i in range(20):
    #     b.generate_centers()
    #     b.generate_blocks()
    #     b.save_block_map()