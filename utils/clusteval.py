import numpy as np
import json
from nltk.metrics import recall, precision, f_measure
from nltk.metrics.distance import edit_distance
from sklearn.metrics import roc_curve 
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DEBUG = False
NULL = 'NULL'
END = '</s>'
EPS = 1e-17

#logging.basicConfig(filename="clusteval.log", format="%(asctime)s %(message)s", level=logging.DEBUG)
#logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

#
# Parameters:
# ----------  
# pred (clusters) --- A list of cluster indices corresponding to each sample
# gold (classes) --- A list of class indices corresponding to each sample
def cluster_confusion_matrix(pred, gold, create_plot=True, alignment=None, file_prefix='cluster_confusion_matrix'):
  assert len(pred) == len(gold) 
  n = len(pred)
  if alignment is not None:
    for i, a in enumerate(alignment):
      pred[i] = np.array(pred[i])[a['alignment']].tolist()

  n_cp, n_cg = 1, 1
  for p, g in zip(pred, gold):
    # XXX: Assume p, q are lists and class indices are zero-based
    for c_p, c_g in zip(p, g):
      if c_g + 1 > n_cg:
        n_cg = c_g + 1
      
      if c_p + 1 > n_cp:
        n_cp = c_p + 1
  cm = np.zeros((n_cp, n_cg))
  
  for p, g in zip(pred, gold):
    for c_p, c_g in zip(p, g):
      cm[c_p, c_g] += 1.

  cm = (cm / np.maximum(np.sum(cm, axis=0), EPS)).T
  print('Cluster purity: ', np.mean(np.max(cm, axis=-1)))
  if create_plot:
    fig, ax = plt.subplots(figsize=(20, 30))
    ax.set_xticks(np.arange(cm.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(cm.shape[0])+0.5, minor=False)
    ax.set_xticklabels([str(c) for c in range(n_cg)], minor=False)
    ax.set_yticklabels([str(c) for c in range(n_cp)], minor=False) 
    plt.pcolor(cm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.savefig(file_prefix, dpi=100)
    plt.close()
  np.savez(file_prefix+'.npz', cm)

def word_cluster_confusion_matrix(pred_info, gold_info, concept2idx=None, file_prefix='audio_confusion_matrix'):
  pred, gold = [], []
  for p, g in zip(pred_info, gold_info):
    pred_assignment = p['concept_alignment']
    gold_alignment = g['alignment']

    i_prev = gold_alignment[0]
    if concept2idx is not None:
      concepts = [concept2idx[c] for c in g['image_concepts']]     
    else:
      concepts = g['image_concepts']

    gold_words = [concepts[i_prev]]
    gold_segmentations = [0]
    # Find the true concept label for each segment
    for i in gold_alignment:
      if i != i_prev:
        gold_words.append(concepts[i])
        gold_segmentations.append(i)
        i_prev = i
    gold.append(gold_words)

    pred_words = []
    for start, end in zip(gold_segmentations[:-1], gold_segmentations[1:]):
      segment = pred_assignment[start:end] 
      counts = {c:0 for c in list(set(segment))}
      for c in segment:
        counts[c] += 1
      pred_words.append(sorted(counts, key=lambda x:counts[x], reverse=True)[0])
    pred.append(pred_words)
  cluster_confusion_matrix(pred, gold) 

#
# Parameters:
# ----------  
# pred (clusters) --- A list of cluster indices corresponding to each sample
# gold (classes) --- A list of class indices corresponding to each sample
def cluster_purity(pred, gold):
  cp = 0.
  n = 0.

  cm = np.zeros
  for p, g in enumerate(pred, gold):
    n_intersects = [] 
    n_intersects.append(len(set(p).intersection(set(g))))

    cp += max(n_intersects)
    n += len(set(p))

  return cp / n

def boundary_retrieval_metrics(pred, gold, out_file='class_retrieval_scores.txt', max_len=2000, return_results=False, debug=False, print_results=True):
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
   
    if debug:
      print("examples " + str(n_ex)) 
      print("# of frames in predicted alignment and gold alignment: %d %d" % (len(p_ali), len(g_ali))) 
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
  if print_results:
    print('Local alignment recall: ' + str(recall))
    print('Local alignment precision: ' + str(precision))
    print('Local alignment f_measure: ' + str(f_measure))
  if return_results:
    return recall, precision, f_measure

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
    #if DEBUG:
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
def segmentation_retrieval_metrics(pred, gold, tolerance=4):
  assert len(pred) == len(gold)
  n = len(pred)
  prec = 0.
  rec = 0.

  # Local retrieval metrics
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    #print(p, g)
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
    if DEBUG:
      print(i, a_i, start, cur)

  boundaries.append((start, len(alignment)))
  return boundaries

if __name__ == '__main__':
  tasks = [3]
  #------------------------#
  # Cluster Purity Metrics #
  #------------------------#
  if 0 in tasks:
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
  #-------------------#
  # Retrieval Metrics #
  #-------------------#
  if 1 in tasks:
    pred_seg_file = "../data/flickr30k/audio_level/flickr_landmarks_env.npz" 
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
    segmentation_retrieval_metrics(pred_seg, gold_seg)
  if 2 in tasks:
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
  #----------------------------------#
  # Clustering and Alignment Metrics #
  #----------------------------------#
  if 3 in tasks:
    '''
    pred_alignment_files = [
    '../hmm_dnn/exp/jan_18_mscoco_vgg16_momentum0.00_lr0.10000_stepscale5_gaussiansoftmax/image_phone_alignment.json', 
    '../hmm_dnn/exp/jan_20_mscoco_gaussian_momentum0.0_lr0.01_stepscale0.001_twolayer/image_phone_iter=10_alignment.json',
    '../hmm_dnn/exp/jan_18_mscoco_vgg16_momentum0.0_lr0.01_stepscale1_twolayer/image_phone_iter=1_alignment.json',
    '../hmm_dnn/exp/jan_18_mscoco_gaussian_momentum0.0_lr0.1_stepscale5_linear/image_phone_alignment.json',
    '../hmm_dnn/exp/jan_18_mscoco_vgg16_momentum0.0_lr0.0_stepscale5_linear/image_phone_iter=1_alignment.json'
    ]
    
    data_dir = '../data/mscoco/'
    data_info_file = data_dir + 'mscoco_subset_concept_info_power_law.json'
    concept_info_file = data_dir + 'mscoco_subset_concept_counts_power_law.json'
    gold_alignment_with_names_file = data_dir + 'mscoco_gold_alignment_power_law.json'
    gold_alignment_file = data_dir + 'mscoco_gold_alignment_power_law.json'    
    for i, pred_alignment_file in enumerate(pred_alignment_files):
      print(pred_alignment_file)
      with open(pred_alignment_file, 'r') as f:
        pred_info = json.load(f)
        
      with open(gold_alignment_file, 'r') as f:
        gold_info = json.load(f)

      pred, gold = [], []
      for p, g in zip(pred_info, gold_info):
        pred.append(p['image_concepts'])
        gold.append(g['image_concepts'])

      
      cluster_confusion_matrix(gold, pred, file_prefix='image_confusion_matrix')
      cluster_confusion_matrix(gold, pred, alignment=gold_info, file_prefix='audio_confusion_matrix')

      boundary_retrieval_metrics(pred_info, gold_info, debug=False)
    '''     
    # TODO
    pred_alignment_files = [
    '../hmm_dnn/exp/jan_20_flickr_vgg16_top100_momentum0.00_lr0.10000_stepscale1_gaussiansoftmax/image_phone_alignment.json',
    '../hmm_dnn/exp/jan_21_flickr_vgg16_top100_momentum0.00_lr0.01000_stepscale1_linear/image_phone_alignment.json',
    '../hmm_dnn/exp/jan_21_flickr_vgg16_top100_momentum0.00_lr0.01000_stepscale1_two_layers/image_phone_alignment.json' 
    ]
    data_dir = '../data/flickr30k/phoneme_level/'    
    gold_alignment_file = '../data/flickr30k/phoneme_level/flickr30k_no_NULL_top_100_gold_alignment.json'
    concept2idx_file = '../data/flickr30k/concept2idx.json' 

    for i, pred_alignment_file in enumerate(pred_alignment_files):
      print(pred_alignment_file)
      with open(pred_alignment_file, 'r') as f:
        pred_info = json.load(f)
        
      with open(gold_alignment_file, 'r') as f:
        gold_info = json.load(f)

      with open(concept2idx_file, 'r') as f:
        concept2idx = json.load(f)
       
      pred, gold = [], []
      for p, g in zip(pred_info, gold_info):
        pred.append(p['image_concepts'])
        gold.append([concept2idx[c] for c in g['image_concepts']])
 
      cluster_confusion_matrix(gold, pred, file_prefix='image_confusion_matrix')
      cluster_confusion_matrix(gold, pred, alignment=gold_info, file_prefix='audio_confusion_matrix')
      boundary_retrieval_metrics(pred_info, gold_info, debug=False)
