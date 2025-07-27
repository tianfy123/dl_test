# mlp_model.py
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), #第一层
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),# 第二层
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)#输出层
        )

    def forward(self, x):
        return self.model(x)

