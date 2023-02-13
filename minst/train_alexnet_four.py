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


tables = [
    (0, 2, 0.30, 0.15),
    (0, 4, 0.40, 0.05),
    (0, 6, 0.25, 0.20),
    (0, 8, 0.35, 0.10),
    (2, 0, 0.40, 0.15),
    (2, 2, 0.25, 0.05),
    (2, 4, 0.35, 0.20),
    (2, 6, 0.20, 0.10),
    (2, 8, 0.30, 0.00),
    (4, 0, 0.35, 0.05),
    (4, 2, 0.20, 0.20),
    (4, 4, 0.30, 0.10),
    (4, 6, 0.40, 0.00),
    (4, 8, 0.25, 0.15),
    (6, 0, 0.30, 0.20),
    (6, 2, 0.40, 0.10),
    (6, 4, 0.25, 0.00),
    (6, 6, 0.35, 0.15),
    (6, 8, 0.20, 0.05),
    (8, 0, 0.25, 0.10),
    (8, 2, 0.35, 0.00),
    (8, 4, 0.20, 0.15),
    (8, 6, 0.30, 0.05),
    (8, 8, 0.40, 0.20)
]

for a, g, c, l in tables:
    # 定义训练数据集
    train_mnist = MyDataset(
        "../data/minst_four_alexnet_trainset/MINST-{:.0f}-{:.0f}-{:.2f}-{:.2f}-image.pt".format(a, g, c, l),
        "../data/minst_four_alexnet_trainset/MINST-{:.0f}-{:.0f}-{:.2f}-{:.2f}-label.pt".format(a, g, c, l))

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
    net_path = "../models/four_alexnet/four_alexnet_{:.0f}_{:.0f}_{:.2f}_{:.2f}.pt".format(a, g, c, l)
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
                print(a, g, c, l)
                break
            else:
                early_stopping.early_stop = False
                