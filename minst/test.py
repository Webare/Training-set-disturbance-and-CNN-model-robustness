import torch

from load_dataset import MyDataset
from models import AlexNet, LeNet
from load_dataset import setRandomSeed


def test(model, device, test_loader):
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    print("Test Accuracy = {} / {} = {:.6f}".format(correct, total, acc))
    return acc


BATCH_SIZE = 64
setRandomSeed(1)
# 定义是否使用GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = LeNet().to(device)

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
    pretrained_model = "../models/four_lenet/four_lenet_{:.0f}_{:.0f}_{:.2f}_{:.2f}.pt".format(a, g, c, l)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    model.eval()

    total_acc = 0
    for i in range(1, 25):
        test_mnist = MyDataset(
            "../data/minst_gaussian_noise_testset/minst_gaussian_noise_test_images_{:0.2f}.pt".format(i / 2),
            "../data/minst_test_labels.pt")
        test_loader = torch.utils.data.DataLoader(
            test_mnist,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        total_acc += test(model, device, test_loader)
    print(a, g, c, l, total_acc / 24)



