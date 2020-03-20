import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

DEBUG = False
def calc_recalls(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A2I_ind = A2I_ind.data
    I2A_ind = I2A_ind.data
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    #loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    loss = Variable(torch.zeros(1), requires_grad=True)

    if torch.cuda.is_available():
      loss = loss.cuda()

    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    #S = torch.zeros(n, n, device=image_outputs.device)
    S = Variable(torch.zeros(n, n))
    if torch.cuda.is_available():
      S = S.cuda()
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def calc_wer_stat(outputs, labels, nwords):
  """
  Assumes outputs is a (temporal_dim, batchsize, vocab_size) tensor
  Assumes labels is a (batchsize, sequence_len) tensor
  Assumes nwords is a (batchsize,) integer-valued vector
  Returns the word error rate
  """ 
  hyps = ctc_decode(outputs)
  refs = labels.data.numpy()
  seqLens = nwords.data.numpy()
  assert len(hyps) == len(refs)

  wer = 0.
  n = len(hyps) 
  
  dist_total = 0.
  ins_total, dels_total, subs_total, corrs_total = 0., 0., 0., 0.
  for i, (hyp, ref) in enumerate(zip(hyps, refs)):
    dist, ins, dels, subs, corrs = edit_distance(ref[:seqLens[i]], hyp)
    dist_total += dist
    ins_total += ins
    dels_total += dels
    subs_total += subs
    corrs_total += corrs
  
    if DEBUG:
      print('hyp: ', hyp)
      print('ref: ', ref)
  return dist_total, ins_total, dels_total, subs_total, corrs_total

def ctc_decode(outputs):
  """
  Assumes outputs is a (temporal_dim, batch_size, vocab_size) tensor
  Returns the label sequence of the best path 
  """ 
  T = outputs.size(0)
  vocab_size = outputs.size(-1)
  blank = 0
  _, best_paths = torch.max(outputs, dim=-1)
  best_paths = torch.t(best_paths).cpu().data.numpy().tolist()
  hyps = []
  for best_path in best_paths:
    # if DEBUG:
    #   print('best_path: ', best_path)
    hyp = []
    for i in range(T):
      if best_path[i] == blank: 
        continue
      elif i != 0 and best_path[i] == best_path[i-1]:
        continue
      else:
        hyp.append(best_path[i])
    hyps.append(hyp)

  return hyps

# Copied from stanford-ctc (https://github.com/amaas/stanford-ctc)
def edit_distance(ref,hyp):
    """
    Edit distance between two sequences reference (ref) and hypothesis (hyp).
    Returns edit distance, number of insertions, deletions and substitutions to
    transform hyp to ref, and number of correct matches.
    """
    n = len(ref)
    m = len(hyp)

    ins = dels = subs = corr = 0
    
    D = np.zeros((n+1,m+1))

    D[:,0] = np.arange(n+1)
    D[0,:] = np.arange(m+1)
    for i in range(1,n+1):
      for j in range(1,m+1):
        if ref[i-1] == hyp[j-1]:
          D[i,j] = D[i-1,j-1]
        else:
          D[i,j] = min(D[i-1,j],D[i,j-1],D[i-1,j-1])+1

    i=n
    j=m
    while i>0 and j>0:
      if ref[i-1] == hyp[j-1]:
          corr += 1
      elif D[i-1,j] == D[i,j]-1:
          ins += 1
          j += 1
      elif D[i,j-1] == D[i,j]-1:
          dels += 1
          i += 1
      elif D[i-1,j-1] == D[i,j]-1:
          subs += 1
      i -= 1
      j -= 1

    ins += i
    dels += j

    return D[-1,-1],ins,dels,subs,corr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10
