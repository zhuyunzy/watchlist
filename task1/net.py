import torch
from torch import nn,optim
import torch.nn.functional as F


# 定义三层全连接网络
class Model(torch.nn.Module):
    def __init__(self,in_dim=28*28,n_hidden_1=300,n_hidden_2=100,out_dim=10):
        # 输入的维度,第一层网络神经元个数,第二层神经元个数,第三层（输出层）神经元个数
        super(Model,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
        # 全连接,标准化,激活函数
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Conv1 (torch.nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 16, kernel_size=3, stride=1),
                                         nn.BatchNorm2d(16),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(16,32, kernel_size=3, stride=1),
                                         nn.BatchNorm2d(32),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.Conv2d(32,64, kernel_size=3, stride=1),
                                         nn.BatchNorm2d(64),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                         nn.BatchNorm2d(128),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(4 * 4 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(1024, 128),
                                         torch.nn.Linear(128, 10))
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0),-1) # 相当于numpy中reshape(-1代表任何值，后面为尺寸）
        x = self.dense(x)
        return x

