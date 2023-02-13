import torch

# all_perturbed_images = torch.load(r"D:\Users\Zili Wu\PythonProjects\Resnet-Car\fund\data\minst_fgsm_lenet_testset.pt")
# for i in range(25):
#     perturbed_images = [lst[i] for lst in all_perturbed_images]
#     torch.save(perturbed_images, "../data/minst_fgsm_lenet_testset/minst_fgsm_lenet_test_images_{:.2f}.pt".format(i/2))

for i in range(1, 25):
    path = "../data/minst_gaussian_noise_trainset_all/minst_gaussian_noise_train_images_{:0.2f}.pt".format(i / 2)
    images = torch.load(path)
    print(images[-1].shape, type(images[-1]))
    for j in range(len(images)):
        if type(images[j]) == type(images[-1]):
            images[j] = torch.from_numpy(images[j])
    print(images[-1].shape, type(images[-1]))
    torch.save(images, path)


