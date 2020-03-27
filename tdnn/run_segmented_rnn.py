import argparse
import os
import sys
import time
import torch
import torchvision
import torchvision.models as imagemodels
from AudioModels import *
import torchvision.transforms as transforms
from traintest import *
from mscoco_segment_dataset import *
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
parser.add_argument('--dataset', default='mscoco_train', choices=['mscoco_130k', 'mscoco_2k', 'mscoco_train'], help='Data set used for training the model')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--class2id_file', type=str, default=None)
parser.add_argument('--n_class', type=int, default=42)
parser.add_argument('--audio_model', type=str, default='tdnn3', choices=['tdnn3'], help='Acoustic model architecture')
parser.add_argument('--optim', type=str, default='sgd',
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--print_class_accuracy', action='store_true', help='Print accuracy for each image class')
parser.add_argument('--pretrain_model_file', type=str, default=None, help='Pretrained parameters file (used only in feature extraction)')
parser.add_argument('--save_features', action='store_true', help='Save the hidden activations of the neural networks')
parser.add_argument('--date', type=str, default='', help='Date of the experiment')

args = parser.parse_args()

if len(args.date) > 0:
  args.exp_dir = 'exp/%s_%s_%s_lr_%.5f_%s' % (args.audio_model, args.dataset, args.optim, args.lr, args.date)
else:
  args.exp_dir = 'exp/%s_%s_%s_lr_%.5f' % (args.audio_model, args.dataset, args.optim, args.lr)

# TODO
feat_configs = {}

if args.dataset == 'mscoco_train':
  data_path = '/home/lwang114/data/mscoco/audio/train2014/wav/'
  train_segment_file = '../data/mscoco/mscoco_train_phone_segments_balanced_train.txt'
  test_segment_file = '../data/mscoco/mscoco_train_phone_segments_balanced_test.txt'
  args.class2id_file = '../data/mscoco/mscoco_train_phone_segments_phone2id.json'
  with open(args.class2id_file, 'r') as f:
    class2idx = json.load(f)
 
  args.n_class = len(class2idx.keys())
  trainset = MSCOCOSegmentCaptionDataset(data_path, train_segment_file, phone2idx_file=args.class2id_file, feat_configs=feat_configs)
  testset = MSCOCOSegmentCaptionDataset(data_path, test_segment_file, phone2idx_file=args.class2id_file, feat_configs=feat_configs) 
  
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False) 

if args.audio_model == 'tdnn3':
  audio_model = TDNN3(n_class=args.n_class)
else:
  raise NotImplementedError('Audio model not implemented')

train(audio_model, train_loader, test_loader, args)  
