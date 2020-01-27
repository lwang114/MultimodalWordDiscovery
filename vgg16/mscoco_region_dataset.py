import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class MSCOCORegionDataset(Dataset):
  def __init__(self, image_npz_file, label_file, transform=None):
    # Inputs:
    # ------
    #       image_npz_file (string): Path to the npz file with images
    #       label_file (string): Path to the text file with (image key, class labels) in each line 
    #       transform (callable, optional)
    self.transform = transform
    self.images = np.load(image_npz_file)
    self.image_keys = []
    self.class_labels = []
    with open(label_file, 'r') as f:
      for line in f:
        k, c = line.strip().split()
        self.class_labels.append(c)
        self.image_keys.append(k)

    self.class2idx = {c:i for i, c in enumerate(self.class_labels)}

  def __len__(self):
    return len(self.image_keys)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    image = self.images[self.image_keys[idx]]
    label = self.class2idx[self.class_labels[idx]]
    if self.transform:
      image = self.transform(image)
      
    return image, transforms.ToTensor(label) 
