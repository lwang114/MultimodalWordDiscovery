import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve 
from collections import defaultdict
from postprocess import _findPhraseFromPhoneme 

NULL = 'NULL'
END = '</s>'
DEBUG = True
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

  if normalize:
    attention = (attention.T / np.sum(attention, axis=1)).T

  plt.pcolor(attention, cmap=plt.cm.Blues, vmin=0, vmax=1)
  plt.colorbar()
  
  if title:
    plt.title(title)

  if filename:
    plt.savefig(filename, dpi=100)
  else:
    plt.show()
  plt.close()

def plot_img_concept_distribution(json_file, concept2idx_file, out_file='class_distribution', cutoff=100, draw_plot=True):
  labels = []
  with open(json_file, 'r') as f:
    pair_info = json.load(f)

  with open(concept2idx_file, 'r') as f:
    class2idx = json.load(f)
  
  idx2class = sorted(class2idx, key=lambda x:class2idx[x])  

  for p in pair_info:
    concepts = p['image_concepts']
    # Exclude NULL symbol
    labels += [class2idx[c] for c in concepts if c != NULL]
   
  return plot_class_distribution(labels, idx2class, filename=out_file, cutoff=cutoff, draw_plot=draw_plot)

def plot_word_len_distribution(json_file, out_file='word_len_distribution', cutoff=1000, draw_plot=True):
  labels = []
  
  with open(json_file, 'r') as f:
    pair_info = json.load(f)
  
  tot = 0
  for p in pair_info:
    ali = p['alignment']
    sent = p['caption']
    concepts = p['image_concepts']
    phrases, concept_indices = _findPhraseFromPhoneme(sent, ali)
    for i, p in enumerate(phrases):
      if concepts[concept_indices[i]] == NULL or concepts[concept_indices[i]] == END:
        labels.append(len(p))
        tot += 1
    
  if DEBUG:
    print(tot) 
         
  max_len = max(labels)  
  len_dist = np.zeros((max_len+1,))   
  for l in labels:
    len_dist[l] += 1 / tot

  if draw_plot:
    plt.plot(top_classes, top_freqs)
    plt.ylabel('Number of Words')
    plt.xlabel('Word Length')
    if out_file:
      plt.savefig(out_file, dpi=100)
    else:
      plt.show()
    plt.close()
  
  return np.arange(max_len+1), len_dist

def generate_nmt_attention_plots(in_dir, out_dir='', normalize=False):
  for f in filenames:
    fp = open(in_dir + f, 'r')
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
    plot_attention(src_sent, trg_sent, attention, '%s%s.png' % 
                  (out_dir, str(index)), normalize=normalize)

def generate_smt_alignprob_plots(in_file, indices=None, out_dir=''):
  fp = open(in_file, 'r')
  align_info = json.load(fp)
  fp.close()

  for index, ali in enumerate(align_info):
    if indices and index not in indices:
      continue
    sent = ali['caption'] 
    concepts = ali['image_concepts']
    align_prob = np.array(ali['align_probs'])
    normalized = (align_prob.T / np.sum(align_prob, axis=1)).T 
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
  for p, g in zip(pred, gold):
    g_concepts = g['image_concepts'] 
    p_ali, g_ali = p['alignment'], g['alignment']
    p_probs = None
    if 'align_probs' in p:
      p_probs = p['align_probs']
    elif 'attentions' in p:
      p_probs = p['attentions']
    else:
      raise TypeError('Invalid file format')

    for a_p, a_g, p_prob in zip(p_ali, g_ali, p_probs):
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

def plot_avg_roc(pred_json, gold_json, concept2idx='../data/flickr30k/concept2idx.json', freq_cutoff=100, out_file=None):
  top_classes, top_freqs = plot_img_concept_distribution(gold_json, concept2idx, cutoff=10, draw_plot=False)
  #avg_fpr = 0.
  #avg_tpr = 0.
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
  #avg_fpr += fpr
  #avg_tpr += tpr
  
if __name__ == '__main__':
  labels = []
  #top_classes, _ = plot_img_concept_distribution('../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json', '../data/flickr30k/concept2idx.json', cutoff=30)
  '''fig, ax = plt.subplots(figsize=(15, 10))
  top_classes, top_freqs = plot_word_len_distribution('../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json', draw_plot=False)
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
  plt.xlabel('Word Length', size=30) 
  plt.ylabel('Normalized Frequency', size=30)
  plt.legend(labels, prop={'size': 30})  
  plt.savefig('word_len_compare.png')
  plt.close()
  '''
  exp_dir = '../../status_report_mar8th/outputs/samples/'
  pred_json = '../nmt/exp/feb26_normalize_over_time/output/alignment.json' 
  #'../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_alignment.json'
  #'../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json' 
  gold_json = '../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json'
  filenames = os.listdir(exp_dir+'nmt_samples_norm_ov_time/')
  indices = []
  for fn in filenames:
    indices.append(int(fn.split('.')[-2]))

  #'../../status_report_mar8th/samples/nmt_samples_norm_ov_concept/'
  #generate_nmt_attention_plots(exp_dir, exp_dir + '../attention_plots_norm_ov_concept/')
  generate_nmt_attention_plots(exp_dir+'nmt_samples_norm_ov_time/', exp_dir + 'nmt_over_time_plots/', normalize=True)
  #generate_smt_alignprob_plots(pred_json, indices, exp_dir + 'smt_alignment_plots/')
  #generate_gold_alignment_plots(gold_json, indices, exp_dir + 'gold_alignment_plots/')
  #plot_avg_roc(pred_json, gold_json, concept2idx='../data/flickr30k/concept2idx.json', out_file='nmt_roc')
