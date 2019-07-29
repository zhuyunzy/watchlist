import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import ResNet18

# 超参数
learning_rate = 0.01

# 转为张量,标准化
data_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先填充四周,再裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一般的概率翻转，一半不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


# 下载测试集和训练集
trainset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=False)
testset = datasets.CIFAR10(root='./data', train=False, transform=data_tf, download=False)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = ResNet18()
if torch.cuda.is_available():
    model = model.cuda()
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
model.load_state_dict(torch.load('Resnet.pkl'))

# 训练模型
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        print('[%d, %5d] loss: %.4f' % (epoch + 1, (i + 1) *128 , loss.item()))
print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %4f %%' % (
        100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(model.state_dict(), 'Resnet.pkl')


