import argparse
import os
import sys
import time
import torch
import torchvision
import torchvision.models as imagemodels
from ImageModels import *
import torchvision.transforms as transforms
from traintest_vgg16 import *
from mscoco_region_dataset import *
import json
import numpy as np
import random

random.seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--lr_decay', type=int, default=10, help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--dataset', default='mscoco', choices=['mscoco', 'cifar'], help='Data set used for training the model')
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--class2id_file', type=str, default=None)
parser.add_argument('--image_model', type=str, default='vgg16', choices=['vgg16'], help='image model architecture')
parser.add_argument('--optim', type=str, default='sgd',
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--print_class_accuracy', action='store_true', help='Print accuracy for each image class')
args = parser.parse_args()

args.exp_dir = 'exp/%s_%s_lr_%s' % (args.image_model, args.optim, args.lr)
transform = transforms.Compose(
  [transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

'''transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
'''

if args.dataset == 'cifar':
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  class2id = {c:i for i, c in enumerate(classes)}
elif args.dataset == 'mscoco':
  # TODO: extract train and test labels randomly
  data_path = 'data/mscoco/mscoco_subset_130k_images.npz'
  label_file = 'data/mscoco/mscoco_subset_130k_labels.txt'
  train_label_file = 'data/mscoco/mscoco_subset_130k_labels_train.txt'
  test_label_file = 'data/mscoco/mscoco_subset_130k_labels_test.txt'
  
  if not os.path.exists(train_label_file):  
    class_labels = []
    image_keys = []
    with open(label_file, 'r') as f:
      for line in f:
        k, c = line.strip().split()
        class_labels.append(c)
        image_keys.append(k)

    data_npz = np.load(data_path)
    data_keys = data_npz.keys()
    random_indices = np.random.permutation(len(data_keys)).tolist()
    train_keys = [data_keys[i] for i in random_indices[:int(0.8 * len(data_keys))]]
    test_keys = [data_keys[i] for i in random_indices[int(0.8 * len(data_keys)):]]

    with open(train_label_file, 'w') as f:
      for i, k in enumerate(train_keys):
        f.write('%s %s' % (k, class_labels[i]))
      
    with open(test_label_file, 'w') as f:
      for i, k in enumerate(test_keys):
        f.write('%s %s' % (k, class_labels[i]))
     
  trainset = MSCOCORegionDataset(data_path, train_label_file) 
  testset = MSCOCORegionDataset(data_path, test_label_file)   
  
  args.n_class = len(classes)
  if args.class2id_file is None:
    args.class2id_file = '%s_class2idx.json' % args.dataset
  with open(args.class2id_file, 'w') as f:
    json.dump(class2id, f)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

image_model = VGG16(n_class=args.n_class, pretrained=True)

train(image_model, train_loader, test_loader, args)
