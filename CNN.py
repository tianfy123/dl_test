import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

# 归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将像素值转换到[0, 1]
    transforms.Normalize((0.5,), (0.5,))  # 映射到-1到1的范围
])

# 下载训练集和测试集
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 原数据太多，采用抽样数据
# train
sampled_train = torch.randperm(len(trainset))[:7000]  # 抽样索引
sampled_trainset = Subset(trainset, sampled_train)  # 使用Subset创建抽样的数据集
# test
sampled_test = torch.randperm(len(testset))[:3000]
sampled_testset = Subset(testset, sampled_test)

# 创建小批量加载器
train_loader = torch.utils.data.DataLoader(sampled_trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(sampled_testset, batch_size=3000, shuffle=False)

class cnn(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.input_channel = input_channel
        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=(4, 4), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 20 * 20, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 一层卷积
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        # 二层卷积
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        # 全连接层
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def main(epochs):
    # print(111)
    model = cnn(input_channel=1) # 如果是彩色图片input_channel=3
    # 定义优化器
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    # 定义损失函数
    criterion = nn.MSELoss()
    # 开始训练
    model.train()
    # epoch级遍历
    for epoch in range(epochs):
        # 储存一个epoch的损失
        Loss = 0
        # batch级遍历
        for batch_x, batch_y in train_loader:
            batch_y = F.one_hot(batch_y, num_classes=10) # 采用one-hot编码
            out = model.forward(batch_x) # cnn前向传播得到输出
            loss = criterion(batch_y.to(torch.float), out)  # 计算损失
            Loss += loss
            optimizer.zero_grad()# 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step() # 用梯度更新参数
        # 每个epoch完都打印损失值信息
        if epoch % 1 == 0:
            print('Epoch:{}/{}, Loss:{}'.format(epoch + 1, epochs, Loss))

    # 开始预测
    model.eval()
    # 禁用梯度
    with torch.no_grad():
        for tx, ty in test_loader:
            out = model.forward(x=tx)
            pre_y = torch.argmax(out, dim=1)
            accuracy = accuracy_score(pre_y, ty)
            return accuracy


accuracy = main(epochs=5)
print(accuracy)