import torch

images_ori = torch.load("../data/minst_train_images.pt")
labels_ori = torch.load("../data/minst_train_labels.pt")


for i, n in enumerate([5057, 4719, 4381, 4043, 3705, 3367, 3029, 2691]):
    cnt = [0] * 10
    after_cnt = [0] * 10
    images, labels = [], []
    for j in range(len(images_ori)):
        cnt[labels_ori[j]] += 1
        if labels_ori[j] != 1 and cnt[labels_ori[j]] > n:
            continue
        images.append(images_ori[j])
        labels.append(labels_ori[j])
        after_cnt[labels_ori[j]] += 1
    print(cnt)
    print(after_cnt)
    torch.save(images, "../data/minst_class_trainset/MINST-6742-{:0.2f}.pt".format(i / 20 + 0.25))
    torch.save(labels, "../data/minst_class_labels/MINST-6742-{:0.2f}-labels.pt".format(i / 20 + 0.25))
