import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image

class MSCOCORegionDataset(Dataset):
  def __init__(self, image_root_path, bbox_file, class2idx_file, transform=None):
    # Inputs:
    # ------
    #       image_npz_file (string): Path to the npz file with images
    #       label_file (string): Path to the text file with (image key, class labels) in each line 
    #       transform (callable, optional)
    self.transform = transform
    self.image_keys = []
    self.class_labels = []
    self.bboxes = []
    self.image_root_path = image_root_path 
    with open(bbox_file, 'r') as f:
      for line in f:
        k, c, x, y, w, h = line.strip().split()
        self.class_labels.append(c)
        self.image_keys.append('_'.join(k.split('_')[:-1]))
        self.bboxes.append([x, y, w, h])

    with open(class2idx_file, 'r') as f:
      self.class2idx = json.load(f)

  def __len__(self):
    return len(self.image_keys)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    x, y, w, h = self.bboxes[idx]
    # XXX
    x, y, w, h = int(x), int(y), np.maximum(int(w), 1), np.maximum(int(h), 1)
    image = Image.open(self.image_root_path + self.image_keys[idx] + '.jpg').convert('RGB')
    if len(np.array(image).shape) == 2:
      print('Wrong shape')
      image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))  
      image = Image.fromarray(image)
    
    region = image.crop(box=(x, y, x + w, y + h))
    #print(x, y, w, h, region.shape)

    if self.transform:
      region = self.transform(region)
    
    #print(region.mean()) 
    label = self.class2idx[self.class_labels[idx]]
    return region, label
