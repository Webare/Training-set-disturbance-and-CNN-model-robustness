import torch

labels_ori = torch.load("../data/minst_train_labels.pt")

for i in range(1, 11):
    total = i * 330
    cnt = [0] * 10
    after_cnt = [0] * 10
    labels = list()
    for label in labels_ori:
        cnt[label] += 1
        if cnt[label] >= total:
            labels.append(label)
            after_cnt[label] += 1
        else:
            d = cnt[label] // (33 * i)
            labels.append(torch.tensor(d))
            after_cnt[d] += 1
    print(labels[:10], labels[-10:])
    print(len(labels))
    torch.save(labels, "../data/minst_label_labels/MINST-10-{:0.2f}-labels.pt".format(i / 10))

