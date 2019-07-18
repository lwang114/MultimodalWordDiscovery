import numpy as np
import json
from nltk.metrics import recall, precision, f_measure
from nltk.metrics.distance import edit_distance
from sklearn.metrics import roc_curve 
import logging

DEBUG = False
NULL = 'NULL'
END = '</s>'

#logging.basicConfig(filename="clusteval.log", format="%(asctime)s %(message)s", level=logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

#
# :param  pred (clusters) --- A list of cluster indices corresponding to each sample
#         gold (classes) --- A list of class indices corresponding to each sample
#
def local_cluster_purity(pred, gold):
  score = 0
  if DEBUG:
    logging.debug('pred[0]: ' + str(pred[0]))
    logging.debug('gold[0]: ' + str(gold[0]))
    logging.debug('len(pred), len(gold): ' + str(len(pred)) + ' ' + str(len(gold)))
  
  assert len(pred) == len(gold)
  N = len(pred)
  for i, (p, g) in enumerate(zip(pred, gold)):
    if DEBUG:
      logging.debug('pred: ' + str(p))
      logging.debug('gold: ' + str(g))
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

def boundary_retrieval_metrics(pred, gold, out_file='class_retrieval_scores.txt'):
  assert len(pred) == len(gold)
  n = len(pred)
  prec = 0.
  rec = 0.

  # Local retrieval metrics
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    p_ali = p['alignment']
    g_ali = g['alignment']
    v = max(max(set(g_ali)), max(set(p_ali))) + 1
    confusion = np.zeros((v, v))
   
    if DEBUG:
      logging.debug("examples " + str(n_ex)) 
      logging.debug("# of frames in predicted alignment and gold alignment: %d %d" % (len(p_ali), len(g_ali))) 
    assert len(p_ali) == len(g_ali)
    
    for a_p, a_g in zip(p_ali, g_ali):
      confusion[a_g, a_p] += 1.
  
    for i in range(v):
      if confusion[i][i] == 0.:
        continue
      rec += 1. / v * confusion[i][i] / np.sum(confusion[i])   
      prec += 1. / v * confusion[i][i] / np.sum(confusion[:, i])
       
  recall = rec / n
  precision = prec / n
  f_measure = 2. / (1. / recall + 1. / precision)
  print('Local alignment recall: ' + str(recall))
  print('Local alignment precision: ' + str(precision))
  print('Local alignment f_measure: ' + str(f_measure))

  '''def _find_distinct_tokens(data):
    tokens = set()
    for datum in data:
      if 'image_concepts' in datum: 
        tokens = tokens.union(set(datum['image_concepts']))
      elif 'foreign_sent' in datum:
        tokens = tokens.union(set(datum['foreign_sent']))
    return list(tokens)
  
  # Retrieval metrics across classes
  classes = _find_distinct_tokens(gold)
  class2id = {c:i for i, c in enumerate(classes)}
  n_c = len(classes)
  g_confusion = np.zeros((n_c, 2, 2))

  for p, g in zip(pred, gold):
    p_ali = p['alignment']
    g_ali = g['alignment']
    p_concepts = p['image_concepts']
    g_concepts = g['image_concepts']
    for a_p, a_g in zip(p_ali, g_ali):
      for i_c in range(n_c):
        p_c = p_concepts[a_p]
        g_c = g_concepts[a_g]
        g_confusion[class2id[g_c], int(class2id[g_c] == i_c), int(class2id[p_c] == i_c)] += 1
  
  class_recall = np.zeros((n_c,))
  class_precision = np.zeros((n_c,))
  for i in range(n_c):
    class_recall[i] = g_confusion[i, 1, 1] / (g_confusion[i, 1, 1] + g_confusion[i, 1, 0])
    class_precision[i] = g_confusion[i, 1, 1] / (g_confusion[i, 1, 1] + g_confusion[i, 0, 1]) 
  
  print('Average class recall: ', class_recall.mean())
  print('Average class precision: ', class_precision.mean())
  print('Average class f_measure: ', 2 / (1 / class_recall.mean() + 1 / class_precision.mean()))
  with open(out_file, 'w') as f:
    for i in range(n_c):
      f.write('%s %0.4f %0.4f\n' % (classes[i], class_recall[i], class_precision[i]))
  '''

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
  
  logging.info('Precision: ', prec / n)
  logging.info('Recall: ', rec / n)
  logging.info('F measure: ', f_mea / n) 

def accuracy(pred, gold):
  assert len(pred) == len(gold)
  acc = 0.
  n = 0.
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    ali_p = p['alignment']
    ali_g = g['alignment']
    if DEBUG:
      logging.debug("examples " + str(n_ex)) 
      print("examples " + str(n_ex))
      #logging.debug("# of frames in predicted alignment and gold alignment: %d %d" % (len(ali_p), len(ali_g))) 
      print("# of frames in predicted alignment and gold alignment: %d %d" % (len(ali_p), len(ali_g)))

    assert len(ali_p) == len(ali_g)
    for a_p, a_g in zip(ali_p, ali_g):
      acc += (a_p == a_g)
      n += 1
  
  return acc / n

def word_IoU(pred, gold): 
  if DEBUG:
    logging.debug("# of examples in pred and gold: %d %d" % (len(pred), len(gold)))
  assert len(pred) == len(gold)
  def _IoU(pred, gold):
    p_start, p_end = pred[0], pred[1]
    g_start, g_end = gold[0], gold[1]
    i_start, u_start = max(p_start, g_start), min(p_start, g_start)  
    i_end, u_end = min(p_end, g_end), max(p_end, g_end)
  
    if i_start >= i_end:
      return 0.

    if u_start == u_end:
      return 1.

    iou = (i_end - i_start) / (u_end - u_start)
    assert iou <= 1 and iou >= 0
    return iou
 
  iou = 0.
  n = 0.
  for p, g in zip(pred, gold):
    p_word_boundaries = _findWords(p['alignment'])
    g_word_boundaries = _findWords(g['alignment'])
    
    if DEBUG:
      logging.debug("pred word boundaries: " + str(p_word_boundaries))
      logging.debug("groundtruth word boundaries: " + str(g_word_boundaries))
    
    for p_wb in p_word_boundaries: 
      n_overlaps = []
      for g_wb in g_word_boundaries:
        n_overlaps.append(_IoU(g_wb, p_wb))
      max_iou = max(n_overlaps)
      iou += max_iou
      n += 1
  return iou / n

# Boundary retrieval metrics for word segmentation
def segmentation_retrieval_metrics(pred, gold, tolerance):
  assert len(pred) == len(gold)
  n = len(pred)
  prec = 0.
  rec = 0.

  # Local retrieval metrics
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    #print(p, g)
    overlaps = []
    for i, p_wb in enumerate(p.tolist()):
      for g_wb in g.tolist():
        if abs(g_wb[0] - p_wb[0]) <= tolerance and abs(g_wb[1] - p_wb[1]) <= tolerance:
          overlaps.append(i)
    
    rec +=  len(overlaps) / len(g)   
    prec += len(overlaps) / len(p)
           
  recall = rec / n
  precision = prec / n
  if recall <= 0. or precision <= 0.:
    f_measure = 0.
  else:
    f_measure = 2. / (1. / recall + 1. / precision)
  print('Segmentation recall: ' + str(recall))
  print('Segmentation precision: ' + str(precision))
  #print('Segmentation f_measure: ' + str(f_measure))

def _findWords(alignment):
  cur = alignment[0]
  start = 0
  end = 0
  boundaries = []
  for i, a_i in enumerate(alignment):
    if cur == a_i:
      end += 1
    else:
      boundaries.append((start, end))
      start = end
      cur = a_i
  
  if len(boundaries) == 0:
    boundaries.append((start, end))
  return boundaries


'''class Evaluator():
  def __init__(self, pred_file, gold_file):
    self.pred_file = pred_file
    self.gold_file = gold_file 

  def _tokenize(self):
    return nltk.word_tokenize()
'''
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
