import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MovingMNIST(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.filenames = os.listdir(root)
        self.filenames.sort(key=lambda x: int(x.split("_")[1][:-4]))
        self.lim = 150 / 255

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.root, filename)
        images = np.load(filepath).astype(np.float32)
        images = (images - 0) / (255 - 0)
        images[images >= self.lim] = 1
        images[images < self.lim] = 0
        images = torch.tensor(images)        
        X, Y = images[:5, None, :, :], images[[5], :, :]
        return X, Y
    
    def __len__(self):
        return len(self.filenames)
    
if __name__ == "__main__":
    dataset = MovingMNIST("../../../data")
    X, Y = dataset[0]
    print(X.shape, Y.shape)