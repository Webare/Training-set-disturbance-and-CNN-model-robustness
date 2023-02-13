import torch
import matplotlib.pyplot as plt
import numpy as np

for i in range(1, 25):
    path = "E:/data/minst_fgsm_alexnet_testset/minst_fgsm_alexnet_test_images_{:.2f}.pt".format(i / 2)
    images = torch.load(path)
    arrayImg = images[1].numpy()
    arrayShow = np.squeeze(arrayImg, 0)
    plt.imshow(arrayShow, cmap='gray')
    plt.show()