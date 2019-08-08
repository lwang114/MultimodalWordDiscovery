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
# Parameters:
# ----------  
# pred (clusters) --- A list of cluster indices corresponding to each sample
# gold (classes) --- A list of class indices corresponding to each sample
#
def cluster_purity(pred, gold):
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
    #assert len(p) == len(g)
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

def boundary_retrieval_metrics(pred, gold, out_file='class_retrieval_scores.txt', max_len=2000):
  assert len(pred) == len(gold)
  n = len(pred)
  prec = 0.
  rec = 0.

  # Local retrieval metrics
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    p_ali = p['alignment'][:max_len]
    g_ali = g['alignment'][:max_len]
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

def retrieval_metrics(pred, gold, concept2idx, pred_word_cluster_file="pred_clusters.json"):
  assert len(list(gold)) == len(list(pred))
  n = len(list(gold))
  n_c = len(concept2idx.keys())
  
  pred_word_clusters = {}
  confusion_mat = np.zeros((n_c, n_c))
  for i_ex, (g, p) in enumerate(zip(gold, pred)):
    print("alignment %d" % i_ex)
    g_ali, p_ali = g["alignment"], p["alignment"]
    g_c = g["image_concepts"]
    g_word_boundaries = _findWords(g_ali)
    p_word_boundaries = _findWords(p_ali)
    for p_w_b in p_word_boundaries:
      #print(set(p_ali[p_w_b[0]:p_w_b[1]]))
      p_c = p_ali[p_w_b[0]]
      if p_c not in pred_word_clusters:
        pred_word_clusters[p_c] = []
      pred_word_clusters[p_c].append((i_ex, p_w_b))
       
    for g_w_b in g_word_boundaries:
      g_w = g_ali[g_w_b[0]:g_w_b[1]]
      p_w = p_ali[g_w_b[0]:g_w_b[1]]
      for i_g_a, i_p_a in zip(g_w, p_w):
        i_g_c, i_p_c = concept2idx[g_c[i_g_a]], concept2idx[g_c[i_p_a]]
        confusion_mat[i_g_c, i_p_c] += 1.
        
  rec = np.mean(np.diag(confusion_mat) / np.maximum(np.sum(confusion_mat, axis=0), 1.))
  prec = np.mean(np.diag(confusion_mat) / np.maximum(np.sum(confusion_mat, axis=1), 1.))
  if rec <= 0. or prec <= 0.:
    f_mea = 0.
  else:
    f_mea = 2. / (1. / rec + 1. / prec)
  print('Recall: ', rec)
  print('Precision: ', prec)
  print('F measure: ', f_mea) 

  with open(pred_word_cluster_file, "w") as f:
    json.dump(pred_word_clusters, f)

def accuracy(pred, gold, max_len=2000):
  if DEBUG:
    print("len(pred), len(gold): ", len(pred), len(gold))
  assert len(pred) == len(gold)
  acc = 0.
  n = 0.
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    ali_p = p['alignment'][:max_len]
    ali_g = g['alignment'][:max_len]
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
        n_overlaps.append(intersect_over_union(g_wb, p_wb))
      max_iou = max(n_overlaps)
      iou += max_iou
      n += 1
  return iou / n

# Boundary retrieval metrics for word segmentation
def segmentation_retrieval_metrics(pred, gold, tolerance=3):
  assert len(pred) == len(gold)
  n = len(pred)
  prec = 0.
  rec = 0.

  # Local retrieval metrics
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    print(p, g)
    overlaps = 0.
    for i, p_wb in enumerate(p.tolist()[:-1]):
      for g_wb in g.tolist()[:-1]:
        #if abs(g_wb[0] - p_wb[0]) <= tolerance and abs(g_wb[1] - p_wb[1]) <= tolerance:
        iou = intersect_over_union(p_wb, g_wb) 
        if abs(g_wb[1]-p_wb[1]) <= tolerance or iou > 0.5: 
          overlaps += 1.
    
    rec +=  overlaps / len(g)   
    prec += overlaps / len(p)
           
  recall = rec / n
  precision = prec / n
  if recall <= 0. or precision <= 0.:
    f_measure = 0.
  else:
    f_measure = 2. / (1. / recall + 1. / precision)
  print('Segmentation recall: ' + str(recall))
  print('Segmentation precision: ' + str(precision))
  print('Segmentation f_measure: ' + str(f_measure))

def intersect_over_union(pred, gold):
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
 
def _findWords(alignment):
  cur = alignment[0]
  start = 0
  boundaries = []
  for i, a_i in enumerate(alignment):
    if a_i != cur:
      boundaries.append((start, i))
      start = i
      cur = a_i
    print(i, a_i, start, cur)

  boundaries.append((start, len(alignment)))
  return boundaries


if __name__ == '__main__':
  #clsts = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
  #classes = [1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 1]
  '''data_dir = '../data/'
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

  pred_seg_file = "../data/flickr30k/audio_level/flickr_landmarks_combined.npz" 
  gold_seg_file = "../data/flickr30k/audio_level/flickr30k_gold_segmentation_mfcc_htk.npy"
  pred_seg = np.load(pred_seg_file)
  gold_seg = np.load(gold_seg_file, encoding="latin1")
  seg_keys = sorted(pred_seg, key=lambda x:int(x.split('_')[-1]))
  new_pred_seg = [pred_seg[k] for k in seg_keys]
  pred_seg = []
  for seg in new_pred_seg:
    seg_ = []
    for start, end in zip(seg[:-1].tolist(), seg[1:].tolist()):
      seg_.append([start, end])
    pred_seg.append(np.array(seg_))
  np.save("flickr_pred_syllable_segmentation.npy", pred_seg)
  segmentation_retrieval_metrics(pred_seg, gold_seg)'''
  
  a = [0, 0, 1, 1, 1, 3, 2, 2]
  print(_findWords(a))

  pred_align_file = "../comparison_models/exp/aug1_mkmeans/flickr30k_pred_alignment.json"
  gold_align_file = "../data/flickr30k/audio_level/flickr30k_gold_alignment.json"
  concept2idx_file = "../data/flickr30k/concept2idx.json"
  with open(concept2idx_file, "r") as f:
    concept2idx = json.load(f)
  with open(pred_align_file, "r") as f:
    pred_aligns = json.load(f)
  with open(gold_align_file, "r") as f:
    gold_aligns = json.load(f)
  
  retrieval_metrics(pred_aligns, gold_aligns, concept2idx)
