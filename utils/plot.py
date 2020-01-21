import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve 
from collections import defaultdict
#from librosa.display import specshow
try:
  from postprocess import _findPhraseFromPhoneme 
  from clusteval import _findWords, boundary_retrieval_metrics  
except:
  from utils.postprocess import _findPhraseFromPhoneme 
  from utils.clusteval import _findWords, boundary_retrieval_metrics

NULL = 'NULL'
END = '</s>'
DEBUG = False
def plot_class_distribution(labels, class_names, cutoff=100, filename=None, normalize=False, draw_plot=True):
  assert type(labels) == list
  n_c = max(labels)
  if DEBUG:
    print('n classes: ', n_c)
  n_plt = n_c
  if n_c > cutoff:
    n_plt = cutoff
  
  dist = np.zeros((n_c+1,))
  tot = 0.
  for c in labels:
    dist[c] += 1
    tot += 1     
  if normalize:
    dist = dist / tot
  top_indices = np.argsort(dist)[::-1][:n_plt]
  if DEBUG:
    print(np.max(top_indices))
  top_classes = [class_names[i] for i in top_indices]
  dist_to_plt = dist[top_indices]

  if draw_plot: 
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_xticks(np.arange(n_plt), minor=False)
    #ax.set_yticks(np.arange(n_plt) + 0.5, minor=False)
    ax.set_xticklabels(top_classes, rotation=45)
  
    plt.plot(np.arange(n_plt), dist_to_plt)
    plt.ylabel('Class distribtuion')
  
    if filename:
      plt.savefig(filename, dpi=100)
    else:
      plt.show()
    plt.close()

  return top_classes, dist_to_plt

# Modified from xnmt/xnmt/plot_attention.py code
def plot_attention(src_sent, trg_sent, attention, filename=None, title=None,  normalize=False):
  fig, ax = plt.subplots(figsize=(7, 14))
  
  if END not in src_sent and NULL not in trg_sent:
    src_sent += END
    trg_sent += END
  ax.set_xticks(np.arange(attention.shape[1])+0.5, minor=False) 
  ax.set_yticks(np.arange(attention.shape[0])+0.5, minor=False) 
  ax.invert_yaxis()
  
  ax.set_xticklabels(trg_sent)
  ax.set_yticklabels(src_sent)
  for tick in ax.get_xticklabels():
    tick.set_fontsize(30)
    tick.set_rotation(45)

  for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
    #tick.set_rotation(45)

  if normalize:
    attention = (attention.T / np.sum(attention, axis=1)).T

  plt.pcolor(attention, cmap=plt.cm.Blues, vmin=0, vmax=1)
  cbar = plt.colorbar()
  for tick in cbar.ax.get_yticklabels():
    tick.set_fontsize(30)

  if title:
    plt.title(title)

  if filename:
    plt.savefig(filename, dpi=100)
  else:
    plt.show()
  plt.close()

def plot_img_concept_distribution(json_file, concept2idx_file=None, out_file='class_distribution', cutoff=100, draw_plot=True):
  labels = []
  with open(json_file, 'r') as f:
    pair_info = json.load(f)
  
  class2idx = {}
  if concept2idx_file:
    with open(concept2idx_file, 'r') as f:
      class2idx = json.load(f)
  else:
    i_c = 0
    for p in pair_info:
      concepts = p["image_concepts"]
      for c in concepts:
        if c not in class2idx:
          class2idx[c] = i_c
          i_c += 1
  
    with open("concept2idx.json", "w") as f:
      json.dump(class2idx, f, indent=4, sort_keys=True)

  idx2class = sorted(class2idx, key=lambda x:class2idx[x])  

  for p in pair_info:
    concepts = p['image_concepts']
    # Exclude NULL symbol
    labels += [class2idx[c] for c in concepts if c != NULL]
   
  return plot_class_distribution(labels, idx2class, filename=out_file, cutoff=cutoff, draw_plot=draw_plot)

def plot_word_len_distribution(json_file, out_file='word_len_distribution', cutoff=1000, draw_plot=True, phone_level=True):
  labels = []
  
  with open(json_file, 'r') as f:
    pair_info = json.load(f)
  
  tot = 0
  for p in pair_info:
    ali = p['alignment']  
    concepts = sorted(p['image_concepts'])
      
    if phone_level:
      sent = p['caption']
      phrases, concept_indices = _findPhraseFromPhoneme(sent, ali)
      for i, ph in enumerate(phrases):
        if concepts[concept_indices[i]] == NULL or concepts[concept_indices[i]] == END:
          continue
        labels.append(len(ph.split()))
        tot += 1
    else:  
      boundaries = _findWords(ali)

      for start, end in boundaries:       
        if concepts[ali[start]] == NULL or concepts[ali[start]] == END: 
          continue
        labels.append(end - start)
        tot += 1

    if DEBUG:
      print(tot) 
      
  max_len = max(labels)  
  len_dist = np.zeros((max_len+1,))   
  
  for l in labels:
    len_dist[l] += 1. / float(tot)

  plt.plot(np.arange(min(cutoff, max_len))+1, len_dist[1:min(max_len, cutoff)+1])
  plt.xlabel('Word Length')
  plt.ylabel('Number of Words')
  
  if out_file:
    plt.savefig(out_file, dpi=100)
  
  if draw_plot:
    plt.show()
  
  plt.close()
  print("Average word length: ", np.dot(np.arange(len(len_dist))+1, len_dist))
  return np.arange(max_len+1), len_dist

def generate_nmt_attention_plots(align_info_file, indices, out_dir='', normalize=False):
  fp = open(align_info_file, 'r')
  align_info = json.load(fp)
  fp.close()

  for index, att_info in enumerate(align_info):  
    if index not in indices:
      continue   
    src_sent = None
    trg_sent = None
    if 'caption' in att_info: 
      src_sent = att_info['caption']
      trg_sent = att_info['image_concepts']
    else:
      src_sent = att_info['src_sent']
      trg_sent = att_info['trg_sent']

    index = att_info['index']
    attention = np.array(att_info['attentions'])
    plot_attention(src_sent, trg_sent, attention, '%s%s.png' % 
                  (out_dir, str(index)), normalize=normalize)

def generate_smt_alignprob_plots(in_file, indices, out_dir='', T=100, log_prob=False):
  fp = open(in_file, 'r')
  align_info = json.load(fp)
  fp.close()

  for index, ali in enumerate(align_info):
    if index not in indices:
      continue
    concepts = ali['image_concepts']
    if 'align_probs' in ali:
      align_prob = np.array(ali['align_probs'])
      if log_prob:
        align_prob = np.exp((align_prob.T - np.amax(align_prob, axis=1)) / T).T
    elif 'align_scores' in ali:
      align_scores = np.array(ali['align_scores'])
      align_prob = np.exp((align_scores.T - np.amax(align_scores, axis=1)) / T).T

    normalized = (align_prob.T / np.sum(align_prob, axis=1)).T 
    if "caption" in ali.keys():
      sent = ali['caption'] 
    else:
      sent = [str(t) for t in range(len(align_prob))]

    if DEBUG:
      print(normalized, np.sum(normalized, axis=1))
    plot_attention(sent, concepts, normalized, '%s%s.png' % (out_dir, str(index)))

'''def compare_attentions(in_files, indices=None): 
  fig, axes = plt.subplots(1, len(attention_files))  
  for f in attention_files:
    for k, d in enumerate(attention_dirs):
      fp = open(d + f, 'r')
      att_info = json.load(fp)
      fp.close()

      if type(att_info) == list:
        att_info = att_info[0]
   
      src_sent = None
      trg_sent = None
      if 'caption' in att_info: 
        src_sent = att_info['caption']
        trg_sent = att_info['image_concepts']
      else:
        src_sent = att_info['src_sent']
        trg_sent = att_info['trg_sent']

      index = att_info['index']
      attention = np.array(att_info['attentions'])
      plot_attention(src_sent, trg_sent, 
                     attention, 
                     filename='%s%s.png' % (out_dir, str(index)),
                     draw_plot=False, 
                     ax=ax[k], 
                     close=False)
'''

def generate_gold_alignment_plots(in_file, indices=None, out_dir=''):
  fp = open(in_file, 'r')
  align_info = json.load(fp)
  fp.close()

  for index, ali in enumerate(align_info):
    if indices and index not in indices:
      continue
    sent = ali['caption'] 
    concepts = ali['image_concepts']
    alignment = ali['alignment']
    alignment_matrix = np.zeros((len(sent), len(concepts)))
    
    for j, a_j in enumerate(alignment):
      alignment_matrix[j, a_j] = 1
    
    plot_attention(sent, concepts, alignment_matrix, '%s%s.png' % (out_dir, str(index)))

def plot_roc(pred_file, gold_file, class_name, out_file=None, draw_plot=True):
  fp = open(pred_file, 'r')
  pred = json.load(fp)
  fp.close()

  fp = open(gold_file, 'r')
  gold = json.load(fp)
  fp.close()

  y_scores = []
  y_true = []
  if DEBUG:
    print("len(pred), len(gold): ", len(pred), len(gold))

  for i, (p, g) in enumerate(zip(pred, gold)):
    g_concepts = g['image_concepts'] 
    p_ali, g_ali = p['alignment'], g['alignment']

    p_probs = None
    if 'align_probs' in p:
      p_probs = p['align_probs']
    elif 'attentions' in p:
      p_probs = p['attentions']
    elif 'align_scores' in p:
      p_probs = p['align_scores']
    else:
      raise TypeError('Invalid file format')

    #if DEBUG:
    #  print("gold image concepts, pred image concepts", g["image_concepts"], p["image_concepts"])
    #  print("i, index, p_ali.shape, g_ali.shape: ", i, p["index"], np.asarray(p_ali).shape, np.asarray(g_ali).shape)
    #  print("# of concepts for gold, # of concepts for pred: ", len(g_concepts), np.asarray(p_probs).shape[1])

    for a_p, a_g, p_prob in zip(p_ali, g_ali, p_probs):
      #if DEBUG:
      #  print("a_p, a_g, p_prob.shape: ", a_p, a_g, np.asarray(p_prob).shape)
      if g_concepts[a_g] == class_name:
        y_true.append(1)
      else:
        y_true.append(0)
      
      y_scores.append(p_prob[a_g])

  fpr, tpr, thresholds = roc_curve(y_true, y_scores)
  if DEBUG:
    print(thresholds)
  
  if draw_plot:
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    if out_file:
      plt.savefig(out_file)
    else:
      plt.show()
    plt.close()
  
  return fpr, tpr, thresholds

def plot_avg_roc(pred_json, gold_json, concept2idx=None, freq_cutoff=100, out_file=None):
  top_classes, top_freqs = plot_img_concept_distribution(gold_json, concept2idx, cutoff=10, draw_plot=False)
  '''
  avg_fpr = 0.
  avg_tpr = 0.
  '''
  fig, ax = plt.subplots()
    
  for c, f in zip(top_classes, top_freqs):
    if f < freq_cutoff:
      continue
    fpr, tpr, _ = plot_roc(pred_json, gold_json, c, draw_plot=False)
    plt.plot(fpr, tpr)

  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend(top_classes, loc='lower right')

  if out_file:
    plt.savefig(out_file)
  else:
    plt.show() 
  plt.close()
  '''
  avg_fpr += fpr
  avg_tpr += tpr
  '''

def plot_acoustic_features(utterance_idx, audio_dir, feat_dir, out_file=None):
  mfccs = np.load(feat_dir+"flickr_mfcc_cmvn.npz", "r")
  bnfs = np.load(feat_dir+"flickr_bnf_all_src.npz", "r")
  mfcc_keys = sorted(mfccs, key=lambda x:int(x.split('_')[-1]))
  bnf_keys = sorted(bnfs, key=lambda x:int(x.split('_')[-1]))
  
  mfcc = mfccs[mfcc_keys[utterance_idx]]
  bnf = bnfs[bnf_keys[utterance_idx]]
  
  plt.figure(figsize=(12, 8))
  plt.subplot(2, 1, 1)
  specshow(mfcc, y_axis="linear")
  plt.colorbar(format="%+2.0f dB")
  plt.title("MFCC of %s" % mfcc_keys[utterance_idx])
  
  plt.subplot(2, 1, 2)
  specshow(bnf, y_axis="linear")
  plt.colorbar(format="%+2.0f dB")
  plt.title("Bottleneck feature of %s" % bnf_keys[utterance_idx])

  if out_file:
    plt.savefig(out_file)
  else:
    plt.show()
  plt.close()

def plot_F1_score_histogram(pred_file, gold_file, concept2idx_file, draw_plot=False, out_file=None):
  # Load the predicted alignment, gold alignment and concept dictionary 
  with open(pred_file, 'r') as f:
    pred = json.load(f)
     
  with open(gold_file, 'r') as f:
    gold = json.load(f)

  with open(concept2idx_file, 'r') as f:
    concept2idx = json.load(f)

  concept_names = [c for c in concept2idx.keys()]
  '''
  top_classes, top_freqs = plot_img_concept_distribution(gold_json, concept2idx, cutoff=10000, draw_plot=False)
  '''
  n_c = len(concept_names)

  # For each concept, compute a concept F1 score by converting the alignment to a binary vector
  f1_scores = np.zeros((n_c,))
  # XXX
  for i_c, c in enumerate(concept_names):
    pred_c = []
    gold_c = []
    
    for p, g in zip(pred, gold):
      concepts = g['image_concepts']
      concepts = [str(concept) for concept in concepts]
      # Skip if the concept is not in the current image-caption pair
      if str(c) not in concepts:
        continue

      p_ali = p['alignment']
      g_ali = g['alignment'] 
      p_ali_c = []
      g_ali_c = [] 
      for a_p, a_g in zip(p_ali, g_ali):
        '''if DEBUG:
          print("a_p, a_g, p_prob.shape: ", a_p, a_g, np.asarray(p_prob).shape)
        '''
        if concepts[a_g] == c:
          g_ali_c.append(1)
        else:
          g_ali_c.append(0)
        
        if concepts[a_p] == c:
          p_ali_c.append(1)
        else:
          p_ali_c.append(0)
      
      pred_c.append({'alignment': p_ali_c})
      gold_c.append({'alignment': g_ali_c})
      
    # XXX
    if len(pred_c) == 0:
      print('Concept not found')
      continue
    _, _, f1_scores[i_c] = boundary_retrieval_metrics(pred_c, gold_c, return_results=True, print_results=False)
 
  concept_order = np.argsort(-f1_scores)
  print(out_file)
  print('Top.5 discovered concepts:', )
  for c in concept_order.tolist()[:5]:
    print(concept_names[c], f1_scores[c])

  print('Top.5 difficult concepts:', )
  for c in concept_order.tolist()[-5:]:
    print(concept_names[c], f1_scores[c])

  # Compute the F1 histogram
  if draw_plot:
    plt.figure()
    plt.hist(f1_scores, bins='auto', density=True)  
    plt.xlabel('F1 score')
    plt.ylabel('Percent of concepts')
    plt.title('Histogram of concept-level F1 scores')
    if out_file:
      plt.savefig(out_file)
    else:
      plt.show()
  else:
    return f1_scores

if __name__ == '__main__':
  '''
  labels = []
  top_classes, _ = plot_img_concept_distribution('../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json', '../data/flickr30k/concept2idx.json', cutoff=30)
  '''
  tasks = [3, 4, 6]
  #--------------------------------------#
  # Phone-level Word Length Distribution #
  #--------------------------------------#
  if 0 in tasks:
    fig, ax = plt.subplots(figsize=(15, 10))
    top_classes, top_freqs = plot_word_len_distribution('../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json', draw_plot=False, phone_level=True)
    plt.plot(top_classes[:50], top_freqs[:50])
    print(np.sum(top_freqs))
    labels.append('Groundtruth')

    top_classes, top_freqs = plot_word_len_distribution('../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_alignment.json', draw_plot=False)
    plt.plot(top_classes[:50], top_freqs[:50])
    print(np.sum(top_freqs))
    labels.append('SMT')

    top_classes, top_freqs = plot_word_len_distribution('../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json', draw_plot=False)
    plt.plot(top_classes[:50], top_freqs[:50])
    print(np.sum(top_freqs))
    labels.append('NMT (Norm over concepts)')

    top_classes, top_freqs = plot_word_len_distribution('../nmt/exp/feb26_normalize_over_time/output/alignment.json', draw_plot=False)
    plt.plot(top_classes[:50], top_freqs[:50])
    print(np.sum(top_freqs))
    labels.append('NMT (Norm over time)')

    ax.set_xticks(np.arange(0, max(top_classes[:50]), 5))
    for tick in ax.get_xticklabels():
      tick.set_fontsize(20)
    for tick in ax.get_yticklabels():
      tick.set_fontsize(20)
    
    plt.xlabel('Word Length', fontsize=30) 
    plt.ylabel('Normalized Frequency', fontsize=30)
    plt.legend(labels, fontsize=30)  
    plt.savefig('word_len_compare.png')
    plt.close()
  #-----------------------------#
  # Phone-level Attention Plots #
  #-----------------------------#
  if 1 in tasks:
    exp_dir = '../../status_report_mar8th/outputs/samples/'
    trg_idx = 1090
    filenames_ov_time = [exp_dir+'nmt_samples_norm_ov_time/'+fn for fn in os.listdir(exp_dir+'nmt_samples_norm_ov_time/') if fn.split('.')[-2] == str(trg_idx)]
    filenames_ov_concept = [exp_dir+'nmt_samples_norm_ov_concept/'+fn for fn in os.listdir(exp_dir+'nmt_samples_norm_ov_concept/') if fn.split('.')[-2] == str(trg_idx)]
    
    indices = [trg_idx]
    '''
    for fn in filenames_ov_time:
      indices.append(fn.split('.')[-2])
    '''
    generate_nmt_attention_plots(filenames_ov_concept, 'nmt_ov_concept_')
    generate_nmt_attention_plots(filenames_ov_time, 'nmt_ov_time_', normalize=True)
    generate_smt_alignprob_plots(pred_json, indices, 'smt_')
    generate_gold_alignment_plots(gold_json, indices, 'gold_')
  #-----------------------#
  # Phone-level ROC Plots #
  #-----------------------#
  if 2 in tasks:
    pred_json = '../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_alignment.json'
    ''''../nmt/exp/feb26_normalize_over_time/output/alignment.json'
    '../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json' 
    '''
    gold_json = '../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json'
    concept2idx_file = '../data/flickr30k/concept2idx.json'
    plot_avg_roc(pred_json, gold_json, concept2idx=concept2idx_file, out_file='nmt_roc')
  #--------------------------#
  # Phone-level F1 Histogram #
  #--------------------------#
  if 3 in tasks:
    draw_plot = False
    width = 0.08
    colors = 'rcgb'
    pred_jsons = [
      '../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_alignment.json',
      '../hmm/exp/aug_31_flickr/flickr30k_pred_alignment.json', 
      '../nmt/exp/feb26_normalize_over_time/output/alignment.json',
      '../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json'
      ] 
    gold_json = '../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json'
    concept2idx_file = '../data/flickr30k/concept2idx.json'
    
    out_files = ['smt_f1_histogram', 'hmm_f1_histogram', 'nmt_novt_histogram', 'nmt_novc_histogram']
    # XXX
    if not draw_plot:
      fig, ax = plt.subplots()
    
    for i, (pred_json, out_file) in enumerate(zip(pred_jsons, out_files)):
      if draw_plot:
        plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)
      else:
        f1_scores = plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)    
        hist, bins = np.histogram(f1_scores, bins=np.linspace(0., 1., 11), density=False)  
        ax.bar(bins[:-1] + width * (1. / len(pred_jsons) * i - 1. / 2), hist, width / len(pred_jsons), color=colors[i])

    if not draw_plot:
      ax.set_xlabel('F1 score')
      ax.set_ylabel('Number of concepts')
      ax.set_title('Histogram of concept-level F1 scores')
      ax.set_xticks(bins[:-1])
      plt.legend(['Mixture SMT', 'HMM SMT', 'NMT (Time-normalized)', 'NMT (Concept-normalized)'], loc='best')
      plt.savefig('f1_histogram_combined')
      plt.close() 
  #-----------------------------#
  # Image-to-phone F1 Histogram #
  #-----------------------------#
  if 4 in tasks:
    draw_plot = False
    width = 0.08
    # Synthetic features
    pred_jsons = [ 
    '../hmm-dnn/exp/jan_15_mscoco_gaussian_momentum0.00_lr1.00000_stepscale5_gaussiansoftmax/image_phone_alignment.json',
    '../hmm-dnn/exp/jan_18_mscoco_gaussian_momentum0.0_lr0.01_stepscale1_twolayer/image_phone_alignment.json',
    '../hmm-dnn/exp/jan_18_mscoco_gaussian_momentum0.0_lr0.1_stepscale5_linear/image_phone_alignment.json'
    ]    
    
    # XXX
    out_files = ['gaussian_hmm_synthetic_f1_histogram', 'hmm_dnn_synthetic_f1_histogram', 'linear_hmm_synthetic_f1_histogram']
    colors = 'rcgb'
    out_files = ['gaussian_hmm_vgg_f1_histogram', 'hmm_dnn_vgg16_f1_histogram', 'linear_hmm_vgg16_f1_histogram'] 
    gold_json = '../data/mscoco/mscoco_gold_alignment_power_law.json'
    concept2idx_file = '../data/mscoco/concept2idx.json'
    with open('../data/mscoco/concept2idx.json', 'w') as f:
      json.dump({i:i for i in range(65)}, f, indent=4, sort_keys=True)
    
    if not draw_plot:
      fig, ax = plt.subplots()

    for i, (pred_json, out_file) in enumerate(zip(pred_jsons, out_files)):
      if draw_plot:
        plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=True, out_file=out_file)
      else:
        f1_scores = plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)    
        hist, bins = np.histogram(f1_scores, bins=np.linspace(0, 1., 11), density=False)  
        ax.bar(bins[:-1] + width * (1. / len(pred_jsons) * i - 1. / 2), hist, width / len(pred_jsons), color=colors[i])
    
    if not draw_plot:
      ax.set_xlabel('F1 score')
      ax.set_ylabel('Number of concepts')
      ax.set_title('Histogram of concept-level F1 scores')
      ax.set_xticks(bins[:-1])
      ax.set_xticklabels([str('%.1f' % v) for v in bins[:-1].tolist()])
      ax.legend(['Gaussian synthetic', 'NN Synthetic', 'Linear Synthetic'], loc='best')
      plt.savefig('image_phone_mscoco_synthetic_f1_histogram_combined')
      plt.close()

    # VGG 16 Features
    pred_jsons = [
    '../hmm-dnn/exp/jan_18_mscoco_vgg16_momentum0.00_lr0.10000_stepscale5_gaussiansoftmax/image_phone_alignment.json',
    '../hmm-dnn/exp/jan_18_mscoco_vgg16_momentum0.0_lr0.01_stepscale1_twolayer/image_phone_iter=1_alignment.json',
    '../hmm-dnn/exp/jan_18_mscoco_vgg16_momentum0.0_lr0.0_stepscale5_linear/image_phone_iter=1_alignment.json'
    ]
    colors = 'rcgb'
    out_files = ['gaussian_hmm_vgg_f1_histogram', 'hmm_dnn_vgg16_f1_histogram', 'linear_hmm_vgg16_f1_histogram'] 
    gold_json = '../data/mscoco/mscoco_gold_alignment_power_law.json'
    concept2idx_file = '../data/mscoco/concept2idx.json'
    with open('../data/mscoco/concept2idx.json', 'w') as f:
      json.dump({i:i for i in range(65)}, f, indent=4, sort_keys=True)
    
    if not draw_plot:
      fig, ax = plt.subplots()

    for i, (pred_json, out_file) in enumerate(zip(pred_jsons, out_files)):
      if draw_plot:
        plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=True, out_file=out_file)
      else:
        f1_scores = plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)    
        hist, bins = np.histogram(f1_scores, bins=np.linspace(0, 1., 11), density=False)  
        ax.bar(bins[:-1] + width * (1. / len(pred_jsons) * i - 1. / 2), hist, width / len(pred_jsons), color=colors[i])
    
    if not draw_plot:
      ax.set_xlabel('F1 score')
      ax.set_ylabel('Number of concepts')
      ax.set_title('Histogram of concept-level F1 scores')
      ax.set_xticks(bins[:-1])
      ax.set_xticklabels([str('%.1f' % v) for v in bins[:-1].tolist()]) 
      # XXX
      plt.legend(['Gaussian VGG', 'NN VGG', 'Linear VGG'], loc='best')
      plt.savefig('image_phone_mscoco_vgg16_f1_histogram_combined')
      plt.close()

  if 5 in tasks:
    audio_dir = "/home/lwang114/data/flickr_audio/wavs/"
    feat_dir = "../data/flickr30k/audio_level/"
    utterance_idx = 0
    plot_acoustic_features(utterance_idx, audio_dir, feat_dir, "%d_features" % utterance_idx)  
  #--------------------------#
  # Audio-level F1 Histogram #
  #--------------------------#
  if 6 in tasks:
    draw_plot = False
    width = 0.08
    concept2idx_file = '../data/flickr30k/concept2idx.json'
    gold_json = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json'
    colors = 'rcgb'
    # MFCC features
    pred_jsons = [
      '../smt/exp/aug_11_segembed_gmm/flickr30k_pred_alignment_resample.json', 
      '../hmm/exp/aug_14_flickr_mfcc/flickr30k_pred_alignment_resample.json', 
      '../smt/exp/aug_16_gmm_mixture_5_mfcc/flickr30k_pred_alignment_resample.json',
      '../smt/exp/aug_19_kmeans_mixture_5_mfcc/flickr30k_pred_alignment_resample.json'
    ]
    out_files = ['seg_gmm_mfcc_f1_histogram', 'seg_kmeans_mfcc_f1_histogram', 'gmm_mfcc_f1_histogram', 'kmeans_mfcc_f1_histogram', 'seg_hmm_mfcc_f1_histogram']
    
    # XXX
    if not draw_plot:
      fig, ax = plt.subplots()
    
    for i, (pred_json, out_file) in enumerate(zip(pred_jsons, out_files)):
      if draw_plot:
        plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)
      else:
        f1_scores = plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)    
        hist, bins = np.histogram(f1_scores, bins=np.linspace(0., 1., 11), density=False)  
        ax.bar(bins[:-1] + width * (1. / len(pred_jsons) * i - 1. / 2), hist, width / len(pred_jsons), color=colors[i])
    
    if not draw_plot:
      ax.set_xlabel('F1 score')
      ax.set_ylabel('Number of concepts')
      ax.set_title('Histogram of concept-level F1 scores')
      ax.set_xticks(bins[:-1])
      ax.set_xticklabels([str('%.1f' % v) for v in bins[:-1].tolist()]) 
      plt.legend(['Segmental GMM', 'Segmental HMM', 'GMM', 'KMeans'], loc='best')
      plt.savefig('audio_f1_histogram_mfcc_combined')
      plt.close() 
    # Bottleneck features
    pred_jsons = ['../smt/exp/aug_13_segembed_gmm_bn/flickr30k_pred_alignment.json',
      '../hmm/exp/aug_15_flickr_bn/flickr30k_pred_alignment.json',
      '../smt/exp/aug_16_gmm_mixture_5_bn/flickr30k_pred_alignment.json',
      '../smt/exp/aug_19_kmeans_mixture_5_bn/flickr30k_pred_alignment.json'
    ]
    out_files = ['seg_gmm_bn_f1_histogram', 'seg_hmm_bn_f1_histogram', 'gmm_bn_f1_histogram', 'kmeans_bn_f1_histogram']
    
    # XXX
    if not draw_plot:
      fig, ax = plt.subplots()
    
    for i, (pred_json, out_file) in enumerate(zip(pred_jsons, out_files)):
      if draw_plot:
        plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)
      else:
        f1_scores = plot_F1_score_histogram(pred_json, gold_json, concept2idx_file=concept2idx_file, draw_plot=draw_plot, out_file=out_file)    
        hist, bins = np.histogram(f1_scores, bins=np.linspace(0., 1., 11), density=False)  
        ax.bar(bins[:-1] + width * (1. / len(pred_jsons) * i - 1. / 2), hist, width / len(pred_jsons), color=colors[i])
     
    if not draw_plot:
      ax.set_xlabel('F1 score')
      ax.set_ylabel('Percent of concepts')
      ax.set_title('Histogram of concept-level F1 scores')
      ax.set_xticks(bins[:-1])
      ax.set_xticklabels([str('%.1f' % v) for v in bins[:-1].tolist()]) 
      plt.legend(['Segmental GMM', 'Segmental HMM', 'GMM', 'KMeans'], loc='best')
      plt.savefig('audio_f1_histogram_bn_combined')
      plt.close() 
