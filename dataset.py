import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import ipdb


class SADataset(Dataset):
    def __init__(self, data_dir="data", split="train") -> None:
        super().__init__()

        img_train_dir = os.path.join(data_dir, "train")
        img_val_dir = os.path.join(data_dir, "val")
        
        self.datasets = []
        self.datasets_name = []
    
        def generate_dataset(split):
            if split == "train":
              img_dir = img_train_dir
            else:
              img_dir = img_val_dir
            self.datasets = []
            self.datasets_name = []
            f_list = os.listdir(img_dir)
            #print(file_list)
            #print(len(file_list))
            
            for i in file_list:
                #print(os.path.splitext(i)[0])
                img_sub_dir = os.path.join(img_dir, i)
                file_sub_list = os.listdir(img_sub_dir)
                useless, label = os.path.splitext(i)[0].split('_')
                #print(useless)
                for j in sub_file_list:
                  X = np.load(os.path.join(img_sub_dir, j))
                  self.datasets.append((X, label))
                  self.datasets_name.append(j)
            return self.datasets, self.datasets_name

        self.datasets, self.datasets_name = generate_dataset(split)
       
        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop(size=(160, 160)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        img, label = self.datasets[index]
        img = self.transform(img)
    
        return img, label


if __name__ == "__main__":
    dataset = SADataset()
    print(dataset[0])
    print(len(dataset))
