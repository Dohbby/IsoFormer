import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
cv2.setNumThreads(1)
class H5Dataset(Dataset):
    def __init__(self, h5_file):
        super(H5Dataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):  # 通过np.expand_dims方法得到组合的新数据
        with h5py.File(self.h5_file, 'r') as f:
            lr=f['data'][idx]
            gt=f['label'][idx]
            return lr,gt

    def __len__(self):  # 得到数据大小
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])