import numpy as np
import matplotlib.pyplot as plt
import os
from util import del_file
import re


class Triangle:
    def __init__(self, center, radius_max, radius_min):
        self.type = 'triangle'

        # 三角形顶点
        r1 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta1 = np.random.rand() * 2 / 3 * np.pi
        r2 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta2 = np.random.rand() * 2 / 3 * np.pi + 2 / 3 * np.pi
        r3 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta3 = np.random.rand() * 2 / 3 * np.pi + 4 / 3 * np.pi
        self.vertex = np.array([[center[0] + r1 * np.cos(theta1), center[1] + r1 * np.sin(theta1)],
                                [center[0] + r2 * np.cos(theta2), center[1] + r2 * np.sin(theta2)],
                                [center[0] + r3 * np.cos(theta3), center[1] + r3 * np.sin(theta3)]])

        # 三角形边
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
        self.type = 'rectangle'
        # 矩形顶点
        r = radius_min + (radius_max - radius_min) * np.random.rand()
        theta1 = np.random.rand() * 2 * np.pi
        theta2 = theta1 + np.random.rand() * np.pi / 2 + np.pi / 4
        theta3 = theta1 + np.pi
        theta4 = theta2 + np.pi
        self.vertex = np.array([[center[0] + r * np.cos(theta1), center[1] + r * np.sin(theta1)],
                                [center[0] + r * np.cos(theta2), center[1] + r * np.sin(theta2)],
                                [center[0] + r * np.cos(theta3), center[1] + r * np.sin(theta3)],
                                [center[0] + r * np.cos(theta4), center[1] + r * np.sin(theta4)]])

        # 矩形的边
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
        self.type = 'circle'
        r = radius_min + (radius_max - radius_min) * np.random.rand()
        self.vertex = np.array([center[0], center[1], r], dtype=object)
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
        self.block = []
        self.radius_max = radius_max
        self.radius_min = radius_min
        self.low = low
        self.high = high
        self.generate_centers()
        self.generate_blocks()
        self._block_map_dir = None

    def generate_centers(self):
        self.points = []
        self.blocks_center = []
        self.block = []
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
                b = Triangle(center=self.blocks_center[i],
                             radius_max=self.radius_max,
                             radius_min=self.radius_min)
            if shape == 'rectangle':
                b = Rectangle(center=self.blocks_center[i],
                              radius_max=self.radius_max,
                              radius_min=self.radius_min)
            if shape == 'circle':
                b = Circle(center=self.blocks_center[i],
                           radius_max=self.radius_max,
                           radius_min=self.radius_min)

            self.points.append(b.edge)
            self.block.append({"shape": b.type, "vertex": b.vertex})

    def save_block_map(self):
        fold_num = len(os.listdir("block map"))
        path = 'block map/%d/' % (fold_num + 1)
        os.mkdir(path)
        np.savetxt(path + 'block center.txt', self.blocks_center)
        for i in range(len(self.blocks_center)):
            np.savetxt(path + 'points_%d.txt' % i, self.points[i])
            np.savetxt(path + 'vertex_%d,txt' % i, self.block[i]["vertex"])

        with open(path + 'block.txt', "a") as file:
            for i in range(len(self.blocks_center)):
                file.write("{}\n".format(self.block[i]["shape"]))

        fig, ax = plt.subplots()
        ax.cla()
        plt.xlim([-30, 250])
        plt.ylim([-30, 250])
        plt.axis('equal')
        for b in self.block:
            if b["shape"] == 'circle':
                circle = plt.Circle((b['vertex'][0], b['vertex'][1]), b['vertex'][2], fill=True)
                ax.add_artist(circle)
            if b["shape"] == 'rectangle':
                plt.fill(b['vertex'][:, 0], b['vertex'][:, 1], fill=True)
            if b["shape"] == 'triangle':
                plt.fill(b['vertex'][:, 0], b['vertex'][:, 1], fill=True)
        plt.savefig(path + '{}.png'.format(fold_num + 1))

    def plot(self):
        plt.clf()
        for obs in self.points:
            plt.plot(obs[:, 0], obs[:, 1], c='b')
        plt.pause(.1)

    def recover_block_map(self, dir='block map/1/'):
        self._block_map_dir = dir
        self.blocks_center = []
        self.points = []
        self.block = []
        self.blocks_center = np.loadtxt(dir + 'block center.txt')
        with open(dir + 'block.txt', "r") as file:
            lines = file.readlines()
        for i in range(len(self.blocks_center)):
            self.points.append(np.loadtxt(dir + 'points_%d.txt' % i))
            vertex = np.loadtxt(dir + 'vertex_%d,txt' % i)
            self.block.append({"shape": lines[i][:-1], "vertex": vertex})

    def block_map_dir(self):
        return self._block_map_dir


if __name__ == '__main__':
    # del_file('block map')
    b = Block(radius_max=30, radius_min=20, low=20, high=200)
    b.recover_block_map()
    # b.plot()
    # print(1)
    # b.save_block_map()
    # for i in range(20):
    #     b.generate_centers()
    #     b.generate_blocks()
    #     b.save_block_map()
