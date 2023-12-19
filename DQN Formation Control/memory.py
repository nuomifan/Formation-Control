import numpy as np


class Memory:
    def __init__(self, input_dims=12, batch_size=32, memory_size=1000000):
        self.states = np.ones((memory_size, input_dims), dtype=np.float)
        self.actions = np.ones(memory_size, dtype=np.float)
        self.rewards = np.ones(memory_size, dtype=np.float)
        self.next_states = np.ones((memory_size, input_dims), dtype=np.float)
        self.dones = np.ones(memory_size, dtype=np.int)

        # 计数器
        self.mem_ctr = 0
        self.memory_size = memory_size
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.memory_size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.mem_ctr += 1

    def sample(self):
        max_memory = min(self.mem_ctr, self.memory_size)
        indices = np.random.choice(a=max_memory, size=self.batch_size, replace=False, p=None)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.mem_ctr


class Prioritized_memory:
    def __init__(self, input_dims=12, batch_size=32, memory_size=1000000, prob_alpha=0.6):
        self.states = np.ones((memory_size, input_dims), dtype=np.float)
        self.actions = np.ones(memory_size, dtype=np.float)
        self.rewards = np.ones(memory_size, dtype=np.float)
        self.next_states = np.ones((memory_size, input_dims), dtype=np.float)
        self.dones = np.ones(memory_size, dtype=np.int)

        # 概率采样的优先度，prob_alpha为1时完全按照优先度采样，prob_alpha为0时为普通采样
        self.prob_alpha = prob_alpha
        self.priorities = np.zeros(memory_size, dtype=np.float32)
        # 计数器
        self.mem_ctr = 0
        self.memory_size = memory_size
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.memory_size
        if self.mem_ctr > 0:
            self.priorities[index] = self.priorities.max()
        else:
            self.priorities[index] = 1

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.mem_ctr += 1

    def sample(self, beta=0.4):
        # 目前记录数据量
        max_memory = min(self.mem_ctr, self.memory_size)
        prios = self.priorities[0:max_memory]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(a=max_memory, size=self.batch_size, replace=False, p=probs)
        # 更新的幅度，weights越大，说明更新幅度越大
        #
        weights = (self.batch_size * probs[indices]) ** -beta
        weights /= weights.max()

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, prios):
        self.priorities[indices] = prios

    def __len__(self):
        return self.mem_ctr
