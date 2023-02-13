import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from models import LeNet, AlexNet
from load_dataset import setRandomSeed


def show(x):
    x = np.array(x).squeeze()
    plt.imshow(x, cmap='gray')
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

train_mnist = datasets.MNIST("../data/", train=True, transform=transforms.ToTensor(), download=True)
test_mnist = datasets.MNIST("../data/", train=False, transform=transforms.ToTensor(), download=True)
train_images = [train_mnist[i][0] for i in range(len(train_mnist))]
train_lables = [train_mnist[i][1] for i in range(len(train_mnist))]
# test_images = [test_mnist[i][0] for i in range(len(test_mnist))]
# test_lables = [test_mnist[i][1] for i in range(len(test_mnist))]
# torch.save(test_lables, "../data/minst_fgsm_alexnet_test_lables.pt")

model = AlexNet().to(device)
pretrained_model = "../models/alexnet_020.pth"
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()


# FGSM 算法攻击代码
def fgsm_attack(model, image, label, device, eps=0.3):
    image = image.to(device)
    image = torch.unsqueeze(image, 0)
    label = [label]
    label = torch.tensor(label)
    label = label.to(device)
    loss = nn.CrossEntropyLoss()
    # 原图像
    ori_image = image.data

    image.requires_grad = True
    outputs = model(image)

    model.zero_grad()
    cost = loss(outputs, label).to(device)
    cost.backward()
    # 图像 + 梯度得到对抗样本
    adv_images = image + eps * image.grad.sign()
    # 进行下一轮对抗样本的生成。破坏之前的计算图
    image = torch.clamp(adv_images, min=0, max=1)

    return image.cpu().detach().numpy()


def perturbed_image(image, label):
    all_perturb_image_lst = []
    all_perturb_dist_lst = []

    for i in range(1, 70):
        image_p = fgsm_attack(model, image, label, device, i / 100)
        norm_dist = torch.dist(image, torch.from_numpy(image_p), p=2)
        # print(i, norm_dist)
        # if i % 10 == 0:
        #     show(image_p)
        all_perturb_dist_lst.append(norm_dist)
        all_perturb_image_lst.append(image_p)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    ret = [all_perturb_image_lst[find_nearest(all_perturb_dist_lst, i / 2)] for i in range(25)]
    return ret


setRandomSeed(1)
all_perturbed_images = []  # (image_id, severity)
for idx, image in enumerate(train_images): # test_images
    all_perturbed_images.append(perturbed_image(image, train_lables[idx])) # test_lables
    if idx % 100 == 0:
        print(idx, len(train_images)) # test_images
    # show(image)
    if idx % 5000 == 0:
        torch.save(all_perturbed_images, "../data/minst_fgsm_alexnet_trainset_{0}.pt".format(idx))
torch.save(all_perturbed_images, "../data/minst_fgsm_alexnet_trainset.pt") # testset

for i in range(25):
    perturbed_images = [j[i] for j in all_perturbed_images]
    torch.save(perturbed_images, "../data/minst_fgsm_alexnet_train_images_{:.2f}.pt".format(i/2)) # test
