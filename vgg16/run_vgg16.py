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
parser.add_argument('--dataset', default='mscoco_130k', choices=['mscoco_130k', 'mscoco_2k', 'mscoco_train', 'cifar', 'flickr'], help='Data set used for training the model')
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--class2id_file', type=str, default=None)
parser.add_argument('--image_model', type=str, default='vgg16', choices=['vgg16', 'res34'], help='image model architecture')
parser.add_argument('--optim', type=str, default='sgd',
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--random_crop', action='store_true', help='Use random cropping as data augmentation')
parser.add_argument('--print_class_accuracy', action='store_true', help='Print accuracy for each image class')
parser.add_argument('--pretrain_model_file', type=str, default=None, help='Pretrained parameters file (used only in feature extraction)')
parser.add_argument('--save_features', action='store_true', help='Save the hidden activations of the neural networks')
args = parser.parse_args()

args.exp_dir = 'exp/jan_31_%s_%s_%s_lr_%s' % (args.image_model, args.dataset, args.optim, args.lr)
transform = transforms.Compose(
  [transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)
print(args.exp_dir)
tasks = [1]

if args.dataset == 'mscoco_130k' or args.dataset == 'mscoco_2k':
  data_path = '/home/lwang114/data/mscoco/val2014/'
  args.class2id_file = 'mscoco_class2id.json'
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)  
  args.n_class = len(class2idx.keys())
elif args.dataset == 'mscoco_train':
  data_path = '/home/lwang114/data/mscoco/train2014/'
  args.class2id_file = 'mscoco_class2id.json'
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)  
  args.n_class = len(class2idx.keys())
# TODO
elif args.dataset == 'flickr':
  data_path = '/home/lwang114/data/flickr'

#------------------#
# Network Training #
#------------------#
if 0 in tasks:
  if args.dataset == 'cifar':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class2id = {c:i for i, c in enumerate(classes)}
  elif args.dataset == 'mscoco_130k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced.txt'
    train_label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced_train.txt'
    test_label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced_test.txt'
    args.class2id_file = 'mscoco_class2id.json'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
    args.n_class = len(class2idx.keys())
    trainset = MSCOCORegionDataset(data_path, train_label_file, class2idx_file=args.class2id_file, transform=transform) 
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform)   
  elif args.dataset == 'mscoco_train':
    data_path = '/home/lwang114/data/mscoco/'
    train_label_file = '../data/mscoco/mscoco_image_subset_260k_image_bboxes_balanced.txt'
    test_label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes_balanced.txt'
    args.class2id_file = 'mscoco_class2id.json'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
    args.n_class = len(class2idx.keys())
    if args.random_crop:
      transform_train = transforms.Compose(
        [transforms.RandomSizedCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )        
    else:
      transform_train = transform
    
    trainset = MSCOCORegionDataset(data_path + 'train2014/', train_label_file, class2idx_file=args.class2id_file, transform=transform_train) 
    testset = MSCOCORegionDataset(data_path + 'val2014/', test_label_file, class2idx_file=args.class2id_file, transform=transform)   
  elif args.dataset == 'mscoco_2k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    train_label_file = '/home/lwang114/data/mscoco/mscoco_image_subset_image_bboxes_balanced_train.txt'
    test_label_file = '/home/lwang114/data/mscoco/mscoco_image_subset_image_bboxes_balanced_test.txt'
    args.class2id_file = 'mscoco_class2id.json'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
    args.n_class = len(class2idx.keys())
    if args.random_crop:
      transform_train = transforms.Compose(
        [transforms.RandomSizedCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )        
    else:
      transform_train = transform
    
    trainset = MSCOCORegionDataset(data_path, train_label_file, class2idx_file=args.class2id_file, transform=transform_train) 
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform)   
  # TODO
  elif args.dataset == 'flickr':


  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
  
  if args.image_model == 'vgg16':
    image_model = VGG16(n_class=args.n_class, pretrained=True)
  elif args.image_model == 'res34':
    image_model = Resnet34(n_class=args.n_class, pretrained=True) 

  train(image_model, train_loader, test_loader, args)

#--------------------------#
# Image Feature Extraction #
#--------------------------#
if 1 in tasks:
  if args.pretrain_model_file is not None:
    pretrain_model_file = args.pretrain_model_file
  else:
    pretrain_model_file = 'exp/vgg16_mscoco_train_sgd_lr_0.001/image_model.1.pth'
  
  if args.dataset == 'mscoco_130k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    args.class2id_file = 'mscoco_class2id.json'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
  
    args.n_class = len(class2idx.keys())
    print(args.n_class)
    test_label_file = '../data/mscoco/mscoco_subset_130k_image_bboxes.txt'
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform) 
  elif args.dataset == 'mscoco_2k':
    data_path = '/home/lwang114/data/mscoco/val2014/'
    args.class2id_file = 'mscoco_class2id.json'
    with open(args.class2id_file, 'r') as f:
      class2idx = json.load(f)
  
    args.n_class = len(class2idx.keys())
    print(args.n_class)
    test_label_file = '../data/mscoco/mscoco_subset_power_law_bboxes.txt'
    testset = MSCOCORegionDataset(data_path, test_label_file, class2idx_file=args.class2id_file, transform=transform) 
  # TODO
  elif args.dataset == 'flickr':

  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
  
  if args.image_model == 'res34':
    image_model = Resnet34(args.n_class, pretrained=True)
  else:
    image_model = VGG16(n_class=args.n_class, pretrained=True)
    image_model.load_state_dict(torch.load(pretrain_model_file))
  args.save_features = True 
  validate(image_model, test_loader, args)

#-------------------------------------------#
# Image Model Pretrained Weights Extraction #
#-------------------------------------------#
if 2 in tasks:
  if args.image_model == 'res34':
    if args.pretrain_model_file is None:
      args.pretrain_model_file = 'exp/jan_31_res34_mscoco_train_sgd_lr_0.001/image_model.10.pth'
    image_model = Resnet34(n_class=args.n_class, pretrained=True)
    image_model.load_state_dict(torch.load(args.pretrain_model_file))
    
    weight_dict = {'weight': image_model.fc.weight.cpu().detach().numpy(),
                   'bias': image_model.fc.bias.cpu().detach().numpy()}
    np.savez('%s/classifier_weights.npz' % args.exp_dir, **weight_dict)
  if args.image_model == 'vgg16':
    image_model = VGG16(n_class=args.n_class, pretrained=True)
    if args.pretrain_model_file is None:
      args.pretrain_model_file = 'exp/vgg16_mscoco_train_sgd_lr_0.001/image_model.1.pth' 
    image_model.load_state_dict(torch.load(args.pretrain_model_file))
    weight_dict = {}
    i = 0
    for child in list(image_model.classifier.children())[-6:]:
      for p in child.parameters():
        print(p.size()) 
        weight_dict['arr_'+str(i)] = p.cpu().detach().numpy()
        i += 1
    np.savez('%s/classifier_weights.npz' % args.exp_dir, **weight_dict)
