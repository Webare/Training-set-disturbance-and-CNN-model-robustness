import numpy as np
import torch
from torch.utils.data import Dataset


def setRandomSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyDataset(Dataset):

    def __init__(self, images_file, labels_file):
        images = torch.load(images_file)
        labels = torch.load(labels_file)
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
