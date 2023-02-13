import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from pytorchtools import EarlyStopping
from models import LeNet, AlexNet
from load_dataset import setRandomSeed, MyDataset

setRandomSeed(1)
# 定义是否使用GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 超参数设置
EPOCH = 100  # 遍历数据集次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)一次训练的样本数，相当于一次将64张图送入
LR = 0.001  # 学习率

# 定义数据预处理方式，将图片转换成张量的形式,因为后续的操作都是以张量形式进行的
transform = transforms.ToTensor()

for idx in range(1, 11):
    # 定义训练数据集
    train_mnist = MyDataset("../data/minst_train_images.pt",
                            "../data/minst_label_labels/MINST-10-{:0.2f}-labels.pt".format(idx / 10))
    # 原始测试集
    test_mnist = MyDataset("../data/minst_test_images.pt", "../data/minst_test_labels.pt")

    # 定义训练批处理数据
    train_loader = torch.utils.data.DataLoader(
        train_mnist,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_mnist,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 定义损失函数loss function 和优化方式（采用SGD）
    net = AlexNet().to(device)
    net_path = "../models/label_alexnet/label_alexnet_{:0.2f}".format(idx / 10)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # 梯度下降法求损失函数最小值
    early_stopping = EarlyStopping(patience=10, verbose=True, path=net_path)

    # 训练
    if __name__ == "__main__":
        # 遍历训练
        for epoch in range(EPOCH):
            sum_loss = 0.0
            correct = 0
            total = 0
            # 读取下载的数据集
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # forward + backward正向传播以及反向传播更新网络参数
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的训练集准确率为：%f' % (epoch + 1, correct / total))
            # 每跑完一次epoch测试一下准确率
            with torch.no_grad():
                correct1 = 0
                total1 = 0
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    # 取得分最高的那个类
                    _, predicted = torch.max(outputs.data, 1)
                    total1 += labels.size(0)
                    correct1 += (predicted == labels).sum()
                print('第%d个epoch的测试集准确率为：%f' % (epoch + 1, correct1 / total1))
            early_stopping(loss, net)
            if early_stopping.early_stop and correct1 / total1 > 0.7:
                print('Early Stopping')
                print(idx)
                break
            else:
                early_stopping.early_stop = False
