import time
import shutil
from util import *
# from warpctc_pytorch import CTCLoss
import torch
import torch.nn as nn
import numpy as np
import sys
import json
import os

# TODO: Greedy layerwise training
def train(audio_model, train_loader, test_loader, args, device_id=0): 
  if torch.cuda.is_available():
    audio_model = audio_model.cuda()
  
  # Set up the optimizer
  # XXX
  '''  
  for p in audio_model.parameters():
    if p.requires_grad:
      print(p.size())
  '''
  trainables = [p for p in audio_model.parameters() if p.requires_grad]
  
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

  audio_model.train()

  running_loss = 0.
  best_acc = 0.
  for epoch in range(args.n_epoch):
    running_loss = 0.
    # XXX
    # adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
    begin_time = time.time()
    audio_model.train()
    for i, audio_input in enumerate(train_loader):
      # XXX
      #if i > 3:
      #  break

      inputs, labels, nframes, nphones = audio_input 
      B = labels.size(0)
      labels_1d = []
      for b in range(B):
        labels_1d += labels[b].data.numpy().tolist()
      labels_1d = torch.IntTensor(labels_1d)
      
      inputs = Variable(inputs)
      labels = Variable(labels_1d)
      nframes = nframes.type(dtype=torch.int)
      nphones = nphones.type(dtype=torch.int)
      
      if torch.cuda.is_available():
        inputs = inputs.cuda()
      
      optimizer.zero_grad()
      outputs = audio_model(inputs)
      # print(nframes.data.numpy())
      # print(nphones.data.numpy())
      # print(inputs.size(), labels.size(), nframes.size(), nphones.size())
      # print(inputs.type(), labels.type(), nframes.type(), nphones.type())

      outputs = outputs.transpose(1, 0).view(-1, outputs.size()[-1])
      # Masked cross entropy loss
      loss = MaskCrossEntropyLoss(outputs, labels, mask) 
      #running_loss += loss.data.cpu().numpy()[0]
      running_loss += loss.data.cpu().numpy()
      loss.backward()
      optimizer.step()
      
      # TODO: Adapt to the size of the dataset
      n_print_step = 200
      if (i + 1) % n_print_step == 0:
        print('Epoch %d takes %.3f s to process %d batches, running loss %.5f' % (epoch, time.time()-begin_time, i, running_loss / n_print_step))
        running_loss = 0.

    print('Epoch %d takes %.3f s to finish' % (epoch, time.time() - begin_time))
    print('Final running loss for epoch %d: %.5f' % (epoch, running_loss / min(len(train_loader), n_print_step)))
    avg_acc = validate(audio_model, test_loader, args)
    
    # Save the weights of the model
    if avg_acc > best_acc:
      best_acc = avg_acc
      if not os.path.isdir('%s' % exp_dir):
        os.mkdir('%s' % exp_dir)

      torch.save(audio_model.state_dict(),
              '%s/audio_model.%d.pth' % (exp_dir, epoch))  
      with open('%s/accuracy_%d.txt' % (exp_dir, epoch), 'w') as f:
        f.write('%.5f' % avg_acc)

def validate(audio_model, test_loader, args):
  if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = nn.DataParallel(audio_model)

  if torch.cuda.is_available():
    audio_model = audio_model.cuda()
  
  n_class = args.n_class
  if args.class2id_file is not None:
    with open(args.class2id_file, 'r') as f:
      class2id = json.load(f)
      classes = [c for c in sorted(class2id, key=lambda x:class2id[x])]
  else:
    classes = [str(i) for i in range(n_class)]

  # audio_model.eval()
  correct = 0.
  error = 0.
  ins_error = 0.
  dels_error = 0.
  subs_error = 0.
  total = 0.

  begin_time = time.time()
  embed1_all = {}
  hyps_all = {}
  with torch.no_grad():  
    for i, audio_input in enumerate(test_loader):
      # XXX
      # print(i)
      # if i < 619:
      #   continue

      audios, labels, nframes, nphones = audio_input
      # XXX
      # print(labels[1].cpu().numpy())
      audios = Variable(audios)
      if torch.cuda.is_available():
        audios = audios.cuda()

      embeds1, outputs = audio_model(audios, save_features=True)
      hyps = ctc_decode(outputs)
      dist, ins, dels, subs, corrs = calc_wer_stat(outputs, labels, nphones)
      error += dist
      ins_error += ins
      dels_error += dels
      subs_error += subs
      correct += corrs 
      total += torch.sum(nphones)

      if args.save_features:
        if args.audio_model == 'blstm2':
          for i_b in range(embeds1.size()[0]):
            feat_id = 'arr_'+str(i * args.batch_size + i_b)
            embed1_all[feat_id] = embeds1[i_b].data.cpu().numpy()     
            hyps_all[feat_id] = [classes[p_idx] for p_idx in hyps[i_b]]    
      n_print_step = 20
      if (i + 1) % n_print_step == 0:
        print('Takes %.3f s to process %d batches, running phone accuracy: %d %%, running phone error rate: %d %%, ins/del/sub error rates: %d %%, %d %%, %d %%' % (time.time()-begin_time, i, 100 * correct / total, 100 * error / total, 100 * ins_error / total, 100 * dels_error / total, 100 * subs_error / total))

    print('Phone error rate of the network: %d %%, %d/%d' % (100 * error / total, error, total))
    
  if not os.path.isdir('%s' % args.exp_dir):
    os.mkdir('%s' % args.exp_dir)

  np.savez(args.exp_dir+'/embed1_all.npz', **embed1_all)   
  with open(args.exp_dir+'/hyps_all.txt', 'w') as f:
    for feat_id in sorted(hyps_all, key=lambda x:int(x.split('_')[-1])):
      f.write(' '.join(hyps_all[feat_id]) + '\n')

  return  100 * correct / total
