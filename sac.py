from torch.nn import Sequential, Linear, ReLU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from memory import Memory


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, in_features=4, hidden_size=128, out_features=2):
        super(SoftQNetwork, self).__init__()
        self.network = Sequential(
            layer_init(Linear(in_features=in_features, out_features=hidden_size)),
            ReLU(),
            layer_init(Linear(in_features=hidden_size, out_features=hidden_size)),
            ReLU(),
            layer_init(Linear(in_features=hidden_size, out_features=hidden_size)),
            ReLU(),
            layer_init(Linear(in_features=hidden_size, out_features=hidden_size)),
            ReLU(),
            layer_init(Linear(in_features=hidden_size, out_features=out_features)),
        )

    def forward(self, state):
        return self.network(state)


class Actor(nn.Module):
    def __init__(self, in_features=4, hidden_size=128, out_features=2):
        super().__init__()

        self.in_features = 4
        self.out_features = out_features
        self.l1 = nn.Sequential(
            layer_init(nn.Linear(in_features, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size))
        )

        self.fc1 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc_logits = layer_init(nn.Linear(hidden_size, out_features))

    def forward(self, x):
        x = torch.Tensor(x)
        x = x.view(-1, self.in_features)
        x = F.relu(self.l1(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action = action.item()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


# Soft Actor Critic
class SoftActorCritic:
    def __init__(self,
                 input_dims=4,
                 hidden_size=128,
                 output_dims=5,
                 learning_rate=1e-3,
                 gamma=0.99,
                 alpha=0.2,
                 replace_target_iter=300,
                 memory_size=100000,
                 batch_size=32):

        # 只考虑最简单的相对偏差输入，因为这样是最接近实际的，其次也降低了复杂度，并且还保持了平移不变性
        # x2-x1, y2-y1
        self.input_dims = input_dims
        # 输出为上、下、左、右不动，五种动作
        self.output_dims = output_dims
        self.hidden_size = hidden_size
        # 学习率，奖励折扣，混乱度因子
        self.lr = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        # 迭代周期，记忆库大小，采样库大小
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        # 网络迭代硬度
        self.tau = 1
        # 计步器
        self.learn_step_counter = 0
        # 初始化记忆库
        self.memory = Memory(input_dims=input_dims, batch_size=batch_size, memory_size=memory_size)
        # 初始化神经网络
        self.actor = Actor(in_features=input_dims, hidden_size=hidden_size, out_features=output_dims)
        self.qf1 = SoftQNetwork(in_features=input_dims, hidden_size=hidden_size, out_features=output_dims)
        self.qf2 = SoftQNetwork(in_features=input_dims, hidden_size=hidden_size, out_features=output_dims)
        self.qf1_target = SoftQNetwork(in_features=input_dims, hidden_size=hidden_size, out_features=output_dims)
        self.qf2_target = SoftQNetwork(in_features=input_dims, hidden_size=hidden_size, out_features=output_dims)

        self.update_network()

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.eps = self.lr
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()),
                                      lr=self.lr, eps=self.eps)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr,
                                          eps=self.eps)

        self.loss_track = []

    def update_network(self, tau=1):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        # 存储信息
        self.memory.remember(state, action, reward, next_state, done)

    def choose_action(self, state):
        action, _, _ = self.actor.get_action(state)
        return action

    def learn(self):
        # 当记忆库不满时不学习
        if len(self.memory) < self.batch_size:
            return

        # 记录学习的次数
        self.learn_step_counter += 1

        # 从所有记忆库中采样批数据
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # CRITIC training
        with torch.no_grad():
            logits = self.actor(next_states)
            policy_dist = Categorical(logits=logits)
            # Action probabilities for calculating the adapted soft-Q loss
            next_state_action_probs = policy_dist.probs
            next_state_log_pi = F.log_softmax(logits, dim=1)

            qf1_next_target = self.qf1_target(next_states)
            qf2_next_target = self.qf2_target(next_states)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * (min_qf_next_target)

        # use Q-values only for the taken actions
        qf1_values = self.qf1(states)
        qf2_values = self.qf2(states)
        index = [i for i in range(32)]
        qf1_a_values = qf1_values[index, actions]
        qf2_a_values = qf2_values[index, actions]
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        logits = self.actor(states)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_pi = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learn_step_counter % 300 == 0:
            # update the target networks
            self.update_network()
