import time
import shutil
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import json
import os

def train(image_model, train_loader, test_loader, args, device_id=0):
  # XXX
  if torch.cuda.is_available():
    image_model = image_model.cuda()
  
  # Set up the optimizer
  '''
  for p in image_model.parameters():
    if p.requires_grad:
      print(p.size())
  '''
  trainables = [p for p in image_model.parameters() if p.requires_grad]
  
  exp_dir = args.exp_dir 
  if args.optim == 'sgd':
    optimizer = torch.optim.SGD(trainables, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
  elif args.optim == 'adam':
    optimizer = torch.optim.Adam(trainables, args.lr,
                        weight_decay=args.weight_decay)
  else:
    raise ValueError('Optimizer %s is not supported' % args.optim)

  image_model.train()

  running_loss = 0.
  best_acc = 0.
  criterion = nn.CrossEntropyLoss()
  for epoch in range(args.n_epoch):
    running_loss = 0.
    adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
    begin_time = time.time()
    image_model.train()
    for i, image_input in enumerate(train_loader):
      # XXX
      #if i > 3:
      #  break

      inputs, labels = image_input 
      inputs = Variable(inputs)
      labels = Variable(labels)
      # XXX
      if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

      optimizer.zero_grad()

      outputs = image_model(inputs)

      # Cross entropy loss
      loss = criterion(outputs, labels) 
      #running_loss += loss.data.cpu().numpy()[0]
      running_loss += loss.data.cpu().numpy()
      loss.backward()
      optimizer.step()
      
      # Adapt to the size of the dataset
      n_print_step = 100
      if (i + 1) % n_print_step == 0:
        print('Epoch %d takes %.3f s to process %d batches, running loss %.5f' % (epoch, time.time()-begin_time, i, running_loss / n_print_step))
        running_loss = 0.

    print('Epoch %d takes %.3f s to finish' % (epoch, time.time() - begin_time))
    print('Final running loss for epoch %d: %.5f' % (epoch, running_loss / min(len(train_loader), n_print_step)))
    avg_acc = validate(image_model, test_loader, args)
    
    # Save the weights of the model
    if avg_acc > best_acc:
      best_acc = avg_acc
      if not os.path.isdir('%s' % exp_dir):
        os.mkdir('%s' % exp_dir)

      torch.save(image_model.state_dict(),
              '%s/image_model.%d.pth' % (exp_dir, epoch))  

def validate(image_model, test_loader, args):
  if torch.cuda.is_available():
    image_model = image_model.cuda()
  
  n_class = args.n_class
  if args.class2id_file is not None:
    with open(args.class2id_file, 'r') as f:
      class2id = json.load(f)
      classes = [c for c in sorted(class2id, key=lambda x:class2id[x])]
  else:
    classes = [str(i) for i in range(n_class)]

  image_model.eval()
  correct = 0.
  total = 0.
  class_correct = [0. for i in range(n_class)]
  class_total = [0. for i in range(n_class)]
  #with torch.no_grad():
  for i, image_input in enumerate(test_loader):
    # XXX
    #if i > 3:
    #  break

    images, labels = image_input
    images = Variable(images)
    labels = Variable(labels)
    if torch.cuda.is_available():
      images = images.cuda()
      labels = labels.cuda()

    outputs = image_model(images) 
    _, predicted = torch.max(outputs.data, 1) 
    total += labels.size(0)
    c = torch.squeeze(predicted == labels.data)
    #print(c.size())

    correct += torch.sum(c)    
    for i_b in range(labels.size(0)):
      #label_idx = labels[i_b].data.cpu().numpy()[0] 
      label_idx = labels[i_b].data.cpu().numpy()
      class_correct[label_idx] += c[i_b].data.cpu().numpy()
      class_total[label_idx] += 1 

  print('Accuracy of the network: %d %%' % (100 * correct / total))
  
  if args.print_class_accuracy:
    for i_c in range(n_class):
      print('Accuracy of %s: %2d %%' % (
        classes[i_c], 100 * class_correct[i_c] / class_total[i_c]))
  
  return  100 * correct / total 
