import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
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
test_images = [test_mnist[i][0] for i in range(len(test_mnist))]
test_lables = [test_mnist[i][1] for i in range(len(test_mnist))]
torch.save(train_images, "../data/minst_trainset.pt")
torch.save(test_images, "../data/minst_testset.pt")
# torch.save(train_lables, "../data/minst_gaussian_noise_train_lables.pt")
# torch.save(test_lables, "../data/minst_gaussian_noise_test_lables.pt")

def gaussian_noise(x, severity=750):
    c = [i / 200000 for i in range(750)][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return x.astype(np.float32)


def perturbed_image(image):
    all_perturb_image_lst = []
    all_perturb_dist_lst = []

    for i in range(1, 750):
        image_p = gaussian_noise(image, i)
        norm_dist = torch.dist(image, torch.from_numpy(image_p), p=2)
        # print(i, norm_dist)
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
for idx, image in enumerate(train_images):
    all_perturbed_images.append(perturbed_image(image))
    if idx % 100 == 0:
        print(idx, len(train_images))
    # show(image)
torch.save(all_perturbed_images, "../data/minst_gaussian_noise_trainset.pt")
for i in range(25):
    perturbed_images = [lst[i] for lst in all_perturbed_images]
    torch.save(perturbed_images, "../data/minst_gaussian_noise_trainset/minst_gaussian_noise_train_images_{:.2f}.pt".format(i/2))

