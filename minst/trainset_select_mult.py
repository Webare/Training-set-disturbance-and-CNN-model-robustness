import torch

ori_images = torch.load("../data/minst_train_images.pt")
ori_labels = torch.load("../data/minst_train_labels.pt")  # [7,2,...]

for adversarial in range(1, 6):
    for gaussian in range(1, 6):
        fgsm_images = torch.load(
            "../data/minst_fgsm_alexnet_trainset_all/minst_fgsm_alexnet_train_images_{:0.2f}.pt"
            .format(2 * adversarial))
        gaussian_images = torch.load(
            "../data/minst_gaussian_noise_trainset_all/minst_gaussian_noise_train_images_{:0.2f}.pt"
            .format(2 * gaussian))
        mult_labels = ori_labels.copy()
        mult_images = ori_images.copy()
        cnts = [0] * 10
        ret_images = []
        ret_labels = []
        for idx, label in enumerate(ori_labels):
            cnts[label] += 1
            if cnts[label] <= 1000:
                ret_images.append(fgsm_images[idx])
                ret_labels.append(label)
            elif cnts[label] <= 2000:
                ret_images.append(gaussian_images[idx])
                ret_labels.append(label)
        print(type(ret_images[0]), type(ret_images[-1]))
        print(ret_images[0].shape, ret_images[-1].shape)

        mult_images.extend(ret_images)
        mult_labels.extend(ret_labels)
        print(type(mult_images[0]))
        print(mult_images[0].shape)
        torch.save(mult_images,
                   "../data/minst_mult_alexnet_trainset/MINST-2000-{:.0f}-{:.0f}-image.pt"
                   .format(2 * adversarial, 2 * gaussian))
        torch.save(mult_labels,
                   "../data/minst_mult_alexnet_trainset/MINST-2000-{:.0f}-{:.0f}-labels.pt"
                   .format(2 * adversarial, 2 * gaussian))
