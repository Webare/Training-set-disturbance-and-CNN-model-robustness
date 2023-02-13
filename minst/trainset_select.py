import torch

ori_labels = torch.load("../data/minst_train_labels.pt")  # [7,2,...]
ori_image = torch.load("../data/minst_train_images.pt")

for i in range(1, 25):
    images = torch.load("../data/minst_fgsm_alexnet_trainset_all/minst_fgsm_alexnet_train_images_{:0.2f}.pt".format(i / 2))
    labels = ori_labels.copy()
    ori_images = ori_image.copy()
    cnts = [0] * 10
    ret_images = []
    ret_labels = []
    for idx, image in enumerate(images):
        cnts[labels[idx]] += 1
        if cnts[labels[idx]] <= 2000:
            ret_images.append(image)
            ret_labels.append(labels[idx])
    print(ret_images[0:100])
    print(ret_labels[0:100])

    ori_images.extend(ret_images)
    labels.extend(ret_labels)
    print(len(ori_images))
    print(len(labels))
    torch.save(ori_images, "../data/minst_fgsm_alexnet_trainset/MINST-20000-{:0.2f}-image.pt".format(i / 2))
    torch.save(labels, "../data/minst_fgsm_alexnet_trainset/MINST-20000-{:0.2f}-labels.pt".format(i / 2))
