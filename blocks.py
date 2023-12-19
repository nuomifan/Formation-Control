import numpy as np
# import cv2
import matplotlib.pyplot as plt
import os


class Triangle:
    def __init__(self, center, radius_max, radius_min):
        r1 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta1 = np.random.rand() * 2 * np.pi
        r2 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta2 = np.random.rand() * 2 * np.pi
        r3 = radius_min + (radius_max - radius_min) * np.random.rand()
        theta3 = np.random.rand() * 2 * np.pi
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
        theta2 = np.random.rand() * 2 * np.pi
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
        self._blocks = []
        self.radius_max = radius_max
        self.radius_min = radius_min
        self.low = low
        self.high = high
        self._blocks_center = []
        self.generate_centers()
        self.generate_blocks()

    def generate_centers(self):
        # 生成障碍物中心，假设车的大小是10cm，障碍物直径30-50cm左右，
        # 假设实验场地是2X2米的范围，那么就是200cm x 200cm，
        # 尝试10000次随机中心，生成尽可能多的障碍物
        center = np.random.randint(self.low, self.high, (1, 2))
        self._blocks_center.append(center)
        for i in range(10000):
            new_center = np.random.randint(self.low, self.high, (1, 2))
            centers = np.vstack(self._blocks_center)
            dis = new_center - centers
            if all((dis[:, 0] ** 2 + dis[:, 1] ** 2) ** 0.5 > self.radius_max * 2):
                self._blocks_center.append(new_center)

        self._blocks_center = np.vstack(self._blocks_center)

    def generate_blocks(self):
        for i in range(len(self._blocks_center)):
            shape = np.random.choice(['triangle', 'rectangle', 'circle'])
            if shape == 'triangle':
                self._blocks.append(Triangle(center=self._blocks_center[i],
                                             radius_max=self.radius_max,
                                             radius_min=self.radius_min).edge)
            if shape == 'rectangle':
                self._blocks.append(Rectangle(center=self._blocks_center[i],
                                              radius_max=self.radius_max,
                                              radius_min=self.radius_min).edge)
            if shape == 'circle':
                self._blocks.append(Circle(center=self._blocks_center[i],
                                           radius_max=self.radius_max,
                                           radius_min=self.radius_min).edge)

    def save_block_map(self):
        fold_num = sum([os.path.isdir(listx) for listx in os.listdir("block map")])
        os.mkdir('block map/%d' % (fold_num + 1))
        np.savetxt('block center.txt', self._blocks_center)
        for i in range(len(self._blocks_center)):
            np.savetxt('block_%d.txt' % i, self._blocks[i])

    def recover_block_map(self, dir='block map/1/'):
        self._blocks_center = np.loadtxt(dir + 'block center.txt')
        for i in range(len(self._blocks_center)):
            self._blocks.append(np.loadtxt(dir + 'block_%d.txt') % i)


if __name__ == '__main__':
    # t = Triangle(np.array([5, 5]), radius_max=30, radius_min=10)
    # r = Rectangle(center=np.array([5, 5]), radius_max=30, radius_min=10)
    b = Block(radius_max=30, radius_min=20)
    for i in b._blocks:
        plt.plot(i.edge[:, 0], i.edge[:, 1])
    plt.pause(.1)
    print(1)
