# coding: utf-8 

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MyError(torch.nn.Module):
    def __init__(self):
        super(MyError, self).__init__()
        
    def forward(self, pred, real):
        return torch.mean(
            torch.abs(torch.sub(pred, real)) / real
        )


class MyDataset(Dataset):
    def __init__(self, cata_type, flag, samples_path):
        self.X_data = np.load(samples_path + "inputs_" + flag + "_" + cata_type + ".npy")
        self.y_data = np.load(samples_path + "labels_" + flag + "_" + cata_type + ".npy")
        
    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, index):
        samples = self.X_data[index, :, :]
        labels = self.y_data[index, :]
        return torch.from_numpy(samples).float(), torch.from_numpy(labels).float()


if __name__ == '__main__':
    """
        the types of catalyst contains {M, M-OH, M-O, M-OOH, OER, ORR, 
        metal, carbon, carbon-metal, binding_energy}
    """

    cata_type = "BE"
    trainset = DataLoader(
        dataset=MyDataset(cata_type=cata_type, flag="train"),
        batch_size=15,
        shuffle=True
        )

    testset = DataLoader(
        dataset=MyDataset(cata_type=cata_type, flag="test"),
        batch_size=14,
        shuffle=True
        )

    for index, data in enumerate(trainset):
        coord, labels = data
        if index == 3:
            break

    print(coord.shape)
    print(labels.shape)