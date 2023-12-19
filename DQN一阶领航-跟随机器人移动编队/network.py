import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU


class Network(nn.Module):
    def __init__(self, input_dims=12, hidden_size=128, output_dims=5):
        super(Network, self).__init__()
        # 包含了智能体和邻居的位置x,y,和速度vx,vy 假设每个智能体均有两个邻居，那么就需要12个输入通道
        # 假设输出是离散的加速度信号，分别表示上下左右和不动，一共5个输出通道
        self.network = Sequential(
            Linear(in_features=input_dims, out_features=hidden_size),
            ReLU(),
            # Linear(in_features=hidden_size, out_features=hidden_size),
            # ReLU(),
            # Linear(in_features=hidden_size, out_features=hidden_size),
            # ReLU(),
            Linear(in_features=hidden_size, out_features=hidden_size),
            ReLU(),
            Linear(in_features=hidden_size, out_features=output_dims),
        )

        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.uniform_(m.bias)
                nn.init.xavier_normal_(m.weight)

    def forward(self, state):
        return self.network(state)
