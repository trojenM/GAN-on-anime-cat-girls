import os 
import torch
from torch.utils.data import Dataset
import cv2

class Catgirls(Dataset):

    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        image = self.data[index]
            
        if self.transform:
            image = self.transform(image)
            return image
        else:
            return None