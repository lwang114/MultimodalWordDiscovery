import numpy as np
import json
from nltk.metrics import recall, precision, f_measure
from nltk.metrics.distance import edit_distance

DEBUG = True
NULL = 'NULL'
END = '</s>'
#
# :param  pred (clusters) --- A list of cluster indices corresponding to each sample
#         gold (classes) --- A list of class indices corresponding to each sample
#
def local_cluster_purity(pred, gold):
  score = 0
  if DEBUG:
    print('pred[0]: ', pred[0])
    print('gold[0]: ', gold[0])
    print('len(pred), len(gold): ', len(pred), len(gold))
  assert len(pred) == len(gold)
  N = len(pred)
  for i, (p, g) in enumerate(zip(pred, gold)):
    if DEBUG:
      print('pred: ', p)
      print('gold: ', g)
    assert len(p) == len(g)
    L = len(p)
    n_g = len(set(g))
    n_p = len(set(p))

    pred_id2pos = {c:i for i, c in enumerate(list(set(p)))}
    gold_id2pos = {c:i for i, c in enumerate(list(set(g)))}
     
    confusion_matrix = np.zeros((n_p, n_g))
    for p_id, g_id in zip(p, g):
      p_pos = pred_id2pos[p_id]
      g_pos = gold_id2pos[g_id]
      confusion_matrix[p_pos, g_pos] += 1
      #print(confusion_matrix)
    score += 1/L * confusion_matrix.max(axis=1).sum()
  return score / N

def cluster_purity(pred, gold):
  cp = 0.
  n = 0.
  for p in pred.values():
    n_intersects = []  
    for g in gold.values():
      n_intersects.append(len(set(p).intersection(set(g))))

    cp += max(n_intersects)
    n += len(set(p))

  return cp / n
   
def retrieval_metrics(pred, gold):
  if not hasattr(pred, 'keys') or not hasattr(gold, 'keys'):
    raise TypeError('pred and gold should be dictionaries')
  assert len(list(gold)) == len(list(pred))
  n = len(list(gold))
  count = 0.
  rec = 0.
  prec = 0.
  f_mea = 0.
  for g, p in zip(gold, pred):
    if g == 'NULL':
      continue
    if not g or not p:
      continue 
    g, p = set(gold[g]), set(pred[p])
    if not g or not p:
      continue
    rec += recall(g, p)
    prec += precision(g, p)
    f_mea += f_measure(g, p)
  
  print('Precision: ', prec / n)
  print('Recall: ', rec / n)
  print('F measure: ', f_mea / n) 

'''class Evaluator():
  def __init__(self, pred_file, gold_file):
    self.pred_file = pred_file
    self.gold_file = gold_file 

  def _tokenize(self):
    return nltk.word_tokenize()
'''
def accuracy(pred, gold):
  assert len(pred) == len(gold)
  acc = 0.
  n = 0.
  assert len(pred) == len(gold)
  for p, g in zip(pred, gold):
    ali_p = p['alignment']
    ali_g = g['alignment']
    assert len(ali_p) == len(ali_g)
    for a_p, a_g in zip(ali_p, ali_g):
      acc += (a_p == a_g)
      n += 1

  return acc / n

'''def edit_distance(pred, gold):
  L_pred = len(pred)
  L_gold = len(gold)
  lev = np.zeros((L_pred, L_gold))
'''
def word_IoU(pred, gold): 
  assert len(pred) == len(gold)
  def _IoU(pred, gold):
    p_start, p_end = pred[1], pred[2]
    g_start, g_end = gold[1], gold[2]
    i_start, u_start = max(p_start, g_start), min(p_start, g_start)  
    i_end, u_end = min(p_end, g_end), max(p_end, g_end)
  
    if i_start >= i_end:
      return 0.

    if u_start == u_end:
      return 1.

    iou = (i_end - i_start) / (u_end - u_start)
    assert iou <= 1 and iou >= 0
    return iou

  def _findPhraseFromPhoneme(sent, alignment):
    if not hasattr(sent, '__len__') or not hasattr(alignment, '__len__'):
      raise TypeError('sent and alignment should be list')
    assert len(sent) == len(alignment)
    cur = alignment[0]
    ws = []
    w_align = []
    w = ''  
    for i, a_i in enumerate(alignment):
      if cur == a_i:
        w = w + ' ' + sent[i]
      else:
        ws.append(w)
        w_align.append(cur)
        w = sent[i]
        cur = a_i
    
    ws.append(w)
    w_align.append(cur)
    return ws
  
  iou = 0.
  n = 0.
  for p, g in zip(pred, gold):
    p_words = _findPhraseFromPhoneme(p['caption'], p['alignment'])
    g_words = _findPhraseFromPhoneme(g['caption'], g['alignment'])
    if DEBUG:
      print(p_words)
      print(g_words)
    
    p_words_with_pos = []
    g_words_with_pos = []
    start = 0
    for p_w in p_words:
      p_words_with_pos.append((p_w, start, start + len(p_w.split())-1))
      start += len(p_w.split())
  
    start = 0
    for g_w in g_words:
      g_words_with_pos.append((g_w, start, start + len(g_w.split())-1))
      start += len(g_w.split())  
    
    for p_w in p_words_with_pos:
      n_overlaps = []
      for g_w in g_words_with_pos:
        n_overlaps.append(_IoU(g_w, p_w))
      max_iou = max(n_overlaps)
      iou += max_iou
      n += 1
  return iou / n


if __name__ == '__main__':
  #clsts = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
  #classes = [1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 1]
  data_dir = '../data/'
  exp_dir = '../smt/exp/ibm1_phoneme_level_clustering/' 
  #'../nmt/exp/feb28_phoneme_level_clustering/output/'
  pred_align_file = exp_dir + 'flickr30k_pred_alignment.json' 
  'alignment.json' 
  #'mscoco/mscoco_val_pred_alignment.json'
  gold_align_file = data_dir + 'flickr30k/phoneme_level/flickr30k_gold_alignment.json'
  #'mscoco/mscoco_val_gold_alignment.json'
  
  clsts = []
  classes = []
  with open(pred_align_file, 'r') as f:   
    clsts_info = json.load(f)
  for c in clsts_info:
    clsts.append(c['alignment'])

  with open(gold_align_file, 'r') as f:
    classes_info = json.load(f) 
  for c in classes_info:
    classes.append(c['alignment'])

  gold_clst_file = data_dir + 'flickr30k/phoneme_level/flickr30k_gold_clusters.json'
  pred_clst_file = exp_dir + 'flickr30k_pred_clusters.json'

  #'cluster.json' 
  
  pred_clsts = []
  gold_clsts = []
  with open(pred_clst_file, 'r') as f:
    pred_clsts = json.load(f)
    
  with open(gold_clst_file, 'r') as f:
    gold_clsts = json.load(f)
   
  print(local_cluster_purity(clsts, classes))
  print(cluster_purity(clsts_info, classes_info))
  retrieval_metrics(pred_clsts, gold_clsts)
