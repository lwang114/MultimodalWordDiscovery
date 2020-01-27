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
import json

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=4)
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
'''transform = transforms.Compose(
  [transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
'''
transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

if args.dataset == 'cifar':
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  class2id = {c:i for i, c in enumerate(classes)}
   
  args.n_class = len(classes)
  if args.class2id_file is None:
    args.class2id_file = '%s_class2idx.json' % args.dataset
  with open(args.class2id_file, 'w') as f:
    json.dump(class2id, f)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

image_model = VGG16(n_class=args.n_class, pretrained=True)

train(image_model, train_loader, test_loader, args)
