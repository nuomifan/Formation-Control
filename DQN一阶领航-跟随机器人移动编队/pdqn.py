# -- coding: utf-8 --
# @Time : 2020/4/28 16:26
# @Author : Chao Zou
# @File : dqn.py
import numpy as np
import random
import torch
import torch.nn as nn
from network import Network
from memory import Prioritized_memory


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self, agent_id,
                 formation,
                 adjacent_matrix,
                 input_dims,
                 hidden_size=128,
                 output_dims=5,
                 learning_rate=1e-3,
                 gamma=0.99,
                 epsilon=0.1,
                 replace_target_iter=300,
                 memory_size=100000,
                 batch_size=32,
                 prob_alpha=0.6):

        self.id = agent_id
        self.formation = formation
        self.A = adjacent_matrix
        # 包含了智能体和邻居的位置x,y,和速度vx,vy 假设每个智能体均有两个邻居，
        # 3个机器人需要12个输入通道
        # 假设输出是离散的加速度信号，分别表示上下左右和不动，一共5个输出通道
        # 机器人编号，输出动作，输入特征
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_size = hidden_size
        # 学习率，损失率，贪婪度
        self.lr = learning_rate
        self.gamma = gamma
        self.eps = epsilon
        # 迭代周期，记忆库大小，采样库大小
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        # 计步器
        self.learn_step_counter = 0
        # 初始化记忆库
        self.memory = Prioritized_memory(input_dims=input_dims, batch_size=batch_size, memory_size=memory_size,
                                         prob_alpha=prob_alpha)
        # 初始化神经网络
        self.eval_net = Network(input_dims=self.input_dims, hidden_size=self.hidden_size,
                                output_dims=self.output_dims)
        self.tar_net = Network(input_dims=self.input_dims, hidden_size=self.hidden_size,
                               output_dims=self.output_dims)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)

        self.loss_track = []

    def store_transition(self, state, action, reward, next_state, done):
        # 存储信息
        self.memory.remember(state, action, reward, next_state, done)

    def heuristic(self, state):
        # 拉式矩阵
        self.L = self.A[self.id, :]
        self.L[self.id] = -sum(self.L)
        self.L = np.kron(self.L, np.eye(2))

        current = np.inner(self.L, state)
        target = np.inner(self.L, self.formation.flatten())

        error = target - current
        if abs(error[0]) > 0.1:
            if error[0] > 0:
                action = 2
            elif error[0] < 0:
                action = 0
        elif abs(error[1]) > 0.1:
            if error[1] > 0:
                action = 3
            elif error[1] < 0:
                action = 1
        else:
            action = 4
        return action

    def choose_action(self, state):
        state = torch.FloatTensor(state).view(1, self.input_dims)
        action_value = self.eval_net(state)
        max_value, action = torch.max(action_value, dim=1)

        if random.random() > self.eps:
            return action.item()
        else:
            return random.randint(a=0, b=4)

        # return action.item() if random.random() > self.eps else random.randint(a=0, b=4)

    def learn(self):
        # 当记忆库不满时不学习
        if len(self.memory) < self.batch_size:
            return

        # 记录学习的次数
        self.learn_step_counter += 1

        # 用Q估计的Q值更新Q目标的Q值
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.tar_net.load_state_dict(self.eval_net.state_dict())
            # print('\ntarget_params_replaced\n')

        # 从所有记忆库中采样批数据
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample()

        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        weights = torch.FloatTensor(weights)

        q_eval = self.eval_net(states)
        q_next = self.tar_net(next_states)
        q_target = q_eval.clone()

        index = [i for i in range(self.batch_size)]
        q_target[index, actions] = rewards + (1 - dones) * q_next.max(dim=1)[0] * self.gamma

        # loss = self.loss_fn(input=q_eval, target=q_target)
        # loss = (q_eval-q_target.detach()).pow(2)*weights

        loss = (q_eval[index, actions] - q_target[index, actions].detach()).pow(2)
        prois = loss + 1e-5
        self.memory.update_priorities(indices, prois.detach().numpy())
        loss = loss * weights
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_track.append(loss.item())

        if self.eps > 0.05:
            self.eps = self.eps * 0.999
        else:
            self.eps = 0.05
