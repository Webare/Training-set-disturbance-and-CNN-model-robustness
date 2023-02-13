import torch

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
    if c > 0.21:
        images_ori = torch.load("../data/minst_class_trainset/MINST-6742-{:0.2f}.pt".format(c))
        labels_ori = torch.load("../data/minst_class_labels/MINST-6742-{:0.2f}-labels.pt".format(c))
    else:
        images_ori = torch.load("../data/minst_train_images.pt")
        labels_ori = torch.load("../data/minst_train_labels.pt")

    labels = list()
    total = int(l * 6600)
    cnt = [0] * 10
    after_cnt = [0] * 10
    for label in labels_ori:
        cnt[label] += 1
        if cnt[label] >= total:
            labels.append(label)
            after_cnt[label] += 1
        else:
            d = cnt[label] // int(660 * l)
            labels.append(torch.tensor(d))
            after_cnt[d] += 1

    fgsm_images = []
    gaussian_images = []
    if a > 0:
        fgsm_images = torch.load(
            "../data/minst_fgsm_alexnet_trainset_all/minst_fgsm_alexnet_train_images_{:0.2f}.pt".format(a))
    if g > 0:
        gaussian_images = torch.load(
            "../data/minst_gaussian_noise_trainset_all/minst_gaussian_noise_train_images_{:0.2f}.pt".format(g))

    images = images_ori
    cnts = [0] * 10
    ret_images = []
    ret_labels = []
    for idx, label in enumerate(labels_ori):
        cnts[label] += 1
        if a > 0 and cnts[label] <= 1000:
            ret_images.append(fgsm_images[idx])
            ret_labels.append(label)
        elif g > 0 and 1000 < cnts[label] <= 2000:
            ret_images.append(gaussian_images[idx])
            ret_labels.append(label)

    images.extend(ret_images)
    labels.extend(ret_labels)

    print(a, g, c, l)
    print(len(images), len(labels))

    torch.save(images, "../data/minst_four_alexnet_trainset/MINST-{:.0f}-{:.0f}-{:.2f}-{:.2f}-image.pt"
               .format(a, g, c, l))
    torch.save(labels, "../data/minst_four_alexnet_trainset/MINST-{:.0f}-{:.0f}-{:.2f}-{:.2f}-label.pt"
               .format(a, g, c, l))
