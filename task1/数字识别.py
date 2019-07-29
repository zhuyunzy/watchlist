import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import net


# 定义超参数
batch_size = 64
learning_rate = 0.0001
n_epochs = 5



# transform.ToTensor()将pytorch处理的对象转为tensor,转换过程中自动将图片标准化,取值范围（0,1）
# Normalize表示减去0.5再除以0.5,将图片转化到-1～1(灰度图）
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])
     # 若为彩色图片则为Normalize([a,b,c],[d,e,f])来表示每个通道的均值和方差

# 下载并加载MNIST手写数字数据集
train_dataset = datasets.MNIST(
    root=' ./data', train=True , transform=data_tf, download=True)
    # 参数：root（根目录）,train(True为训练接，False为测试集）,download(True冲互联网下载数据集,并放在根目录下）
test_dataset = datasets.MNIST(root=' ./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 参数：（加载dataset,shuffle训练时送入的样本数目时打乱数据）
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 导入网络,定义损失函数，优化方法
# model = net.Model()
model = net.Conv1()
if torch.cuda.is_available():
    model = model.cuda()
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
model.load_state_dict(torch.load('conv1.pkl'))

# 训练网络&测试网络
for epoch in range(n_epochs):
    running_loss = 0.0  # 初始化
    running_correct = 0.0
    print("epoch 第{}/{}次 ".format(epoch+1, 5))
    print("-" * 10)
    for data in train_loader:
        train_inputs, train_labels = data
        if torch.cuda.is_available():
            train_inputs, train_labels = Variable(train_inputs.cuda()), Variable(train_labels.cuda())
        else:
            train_inputs, train_labels = Variable(train_inputs), Variable(train_labels)
        # 将数据转为variable模式
        optimizer.zero_grad()  # 优化器梯度初始为0
        # forward
        outputs = model(train_inputs)
        _, pred = torch.max(outputs.data, 1) # 代表在行这个维度,最大的数
        loss = cost(outputs, train_labels)
        # backward
        loss.backward()
        optimizer.step() # 梯度更新
        running_loss += loss.item()
        # loss.item()就是把标量tensor转换为python内置的数值类型
        # 把所有loss值累加
        running_correct += torch.sum(pred == train_labels.data).to(torch.float32)
        # 把所有正确的值累加
    testing_correct = 0
    for data in test_loader:
        test_inputs, test_labels = data
        if torch.cuda.is_available():
            test_inputs, test_labels = Variable(test_inputs.cuda()), Variable(test_labels.cuda())
        else:
            test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
        outputs = model(test_inputs)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == test_labels.data).to(torch.float64)
    print('Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%'
          .format(running_loss/len(test_dataset),
                  100*running_correct/len(train_dataset),
                  100*testing_correct/len(test_dataset)))

torch.save(model.state_dict(),"conv1.pkl") # 保存模型


