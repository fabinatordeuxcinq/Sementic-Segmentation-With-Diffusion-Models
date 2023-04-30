import torch
from glob import glob
import os
import numpy as np
from PIL import Image

PATH  = "/home/fabienpelletier/Documents/DATASETS/dlss21_ho4_data/train"



class MuscleDataset(torch.utils.data.Dataset) :

    def __init__(self, path2data, test_flag=False) :
        super().__init__()
        # path2data is expected to contains 2 folder : images, labels
        X_path = os.path.join(path2data, "images")
        y_path = os.path.join(path2data, "labels")

        # each image is multi modality
        self.e2 = self.__parse__(X_path, "*e2*")
        self.e5 = self.__parse__(X_path, "*e5*")
        self.e8 = self.__parse__(X_path, "*e8*")
        self.y = self.__parse__(y_path, "*")
        print(len(self.e2), len(self.e5), len(self.e8), len(self.y))
        self.test_flag = test_flag
        # they are sorted by name so should be in the good order

    def __split_class__(self, mask) :
        y = np.stack([mask == 0, mask == 1, mask == 2, mask == 3, mask == 4, mask == 5], axis=0)
        return torch.tensor(y,
                            dtype=torch.float16) # float in the smallest resolution

    def __parse__(self, path, reg) :
        files = glob(os.path.join(path, reg))
        return sorted(files, key=os.path.basename)

    def __len__(self,) :
        return len(self.y)

    def __getitem__(self, index) :
        # open example and stack them
        e2 = Image.open(self.e2[index]).resize((128, 128), resample=Image.NEAREST)
        e5 = Image.open(self.e5[index]).resize((128, 128), resample=Image.NEAREST)
        e8 = Image.open(self.e8[index]).resize((128, 128), resample=Image.NEAREST)

        x = np.stack([e2,e5,e8])
        x = torch.tensor(x)
        if self.test_flag :
            # return file id
            fid = os.path.basename(self.e2[index].partition('_e')[0])
            return x, fid
        y = Image.open(self.y[index]).resize((128, 128), resample=Image.NEAREST)
        y = self.__split_class__(np.array(y))
        return x, y






class MuscleDataset2(torch.utils.data.Dataset) :
    """ Same with 2 only 2 modalities and 4 classes to demonstrate that is working in all situations
    """

    def __init__(self, path2data, test_flag=False) :
        super().__init__()
        # path2data is expected to contains 2 folder : images, labels
        X_path = os.path.join(path2data, "images")
        y_path = os.path.join(path2data, "labels")

        # each image is multi modality
        self.e2 = self.__parse__(X_path, "*e2*")
        self.e5 = self.__parse__(X_path, "*e5*")
        self.y = self.__parse__(y_path, "*")
        print(len(self.e2), len(self.e5), len(self.y))
        self.test_flag = test_flag
        # they are sorted by name so should be in the good order

    def __split_class__(self, mask) :
        y = np.stack([mask == 1, mask == 2, mask == 3, mask == 5], axis=0)
        return torch.tensor(y,
                            dtype=torch.float16) # float in the smallest resolution

    def __parse__(self, path, reg) :
        files = glob(os.path.join(path, reg))
        return sorted(files, key=os.path.basename)

    def __len__(self,) :
        return len(self.y)

    def __getitem__(self, index) :
        # open example and stack them
        e2 = Image.open(self.e2[index]).resize((128, 128), resample=Image.ANTIALIAS)
        e5 = Image.open(self.e5[index]).resize((128, 128), resample=Image.ANTIALIAS)
        x = np.stack([e2,e5])
        x = torch.tensor(x)
        if self.test_flag :
            # return file id
            fid = os.path.basename(self.e2[index].partition('_e')[0])
            return x, fid
        y = Image.open(self.y[index]).resize((128, 128), resample=Image.ANTIALIAS)
        y = self.__split_class__(np.array(y))
        return x, y


