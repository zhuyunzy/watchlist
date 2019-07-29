for epoch in range(n_epochs):
    running_loss = 0.0  # 初始化
    running_correct = 0.0
    print("epoch 第{}/{}次 ".format(epoch+1, 10))
    print("-" * 10)
    for data in trainloader:
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
        running_loss += loss.data.item()
        # loss.item()就是把标量tensor转换为python内置的数值类型
        # 把所有loss值累加
        running_correct += torch.sum(pred == train_labels.data).to(torch.float32)
        # 把所有正确的值累加
    model.eval()
    testing_correct = 0
    for data in testloader:
        test_inputs, test_labels = data
        if torch.cuda.is_available():
            test_inputs, test_labels = Variable(test_inputs.cuda()), Variable(test_labels.cuda())
        else:
            test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
        outputs = model(test_inputs)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == test_labels.data).to(torch.float64)
    print('Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%'
          .format(running_loss/len(testset),
                  100*running_correct/len(trainset),
                  100*testing_correct/len(testset)))