from copy import deepcopy
import numpy as np
import logging
import os
import json

DEBUG = False
END = '</s>'
NULL = 'NULL'

if os.path.exists("*.log"):
  os.system("rm *.log")

logging.basicConfig(filename="postprocess.log", format="%(asctime)s %(message)s)", level=logging.DEBUG)

class XNMTPostprocessor():
  def __init__(self, input_dir, is_phoneme=True):
    self.input_dir = input_dir
    self.is_phoneme = is_phoneme

  # Convert the predicted alignment files to the right format for evaluation
  def convert_alignment_file(self, out_file='pred_alignment.json', feat2wav_file=None): 
    all_files = os.listdir(self.input_dir)
    files = []
    for fn in all_files:
      if fn.endswith(".json"):
        files.append(fn)

    files = sorted(files, key=lambda x:int(x.split('.')[-2]))
    alignments = []
    
    feat2wavs = []
    if feat2wav_file:
      with open(feat2wav_file, 'r') as f:
        feat2wavs = json.load(f)
          
    for i, f in enumerate(files):
      if f.split('.')[-1] == 'json':
        fp = open(self.input_dir + f, 'r')
        ali = json.load(fp)
        if type(ali) == list:
          ali = ali[0]

        fp.close()

        if END in ali['src_sent']:
          ali['src_sent'].remove(END)
          # XXX: This will make the file format inconsistent with the one generated by the preprocess.py
          # convert_xnmt_text function; fix this later
          ali['trg_sent'].remove(END)
          ali['alignment'] = ali['alignment'][:-1]
          if NULL not in ali['trg_sent']:
            ali['trg_sent'] = [NULL] + ali['trg_sent']
            ali['alignment'] = [a+1 if a<len(ali['trg_sent'])-1 else 0 for a in ali['alignment']]
          else:
            ali['alignment'] = [a if a<len(ali['trg_sent'])-1 else 0 for a in ali['alignment']]
        
        new_ali = {}
        if not feat2wav_file:
          new_ali = {'index': ali['index'],
                     'is_phoneme': self.is_phoneme,
                     'caption': ali['src_sent'],
                     'image_concepts': ali['trg_sent'],
                     'alignment': ali['alignment'],
                     'attentions': ali['attentions']
                    }
        else:
          feat_len = len(feat2wavs[i])
          ali_len = len(ali['alignment'])
          reduce_factor = ali['reduce_factor']
          new_feat2wav = []
          for i_a in range(ali_len): 
            for i_f in range(feat_len):
              st = feat2wavs[i]['feat2wav'][i_f * reduce_factor][0]
              end = feat2wavs[i]['feat2wav'][max(i_f * reduce_factor + 1, feat_len - 1)][1]
              new_feat2wav.append([st, end])
          
          new_ali = {'index': ali['index'],
                     'is_phoneme': self.is_phoneme,
                     'caption': ali['src_sent'],
                     'image_concepts': ali['trg_sent'],
                     'alignment': ali['alignment'],
                     'attentions': ali['attentions'],
                     'feat2wav': new_feat2wav 
                    }        

        alignments.append(new_ali)
    
    with open(out_file, 'w') as fp:
      json.dump(alignments, fp, indent=4, sort_keys=True)
  
  def convert_retrieval_file(self, in_file, out_file='retrieval_results.txt'):
    fp = open(in_file)
    ret_indices = []
    for line in fp:
      idx_scores = line.strip().split('), (')
      top_indices = []
      for idx_score in idx_scores:
        idx_score = idx_score.replace('[', '')
        idx_score = idx_score.replace('(', '')
        idx_score = idx_score.replace(')', '')
        idx_score = idx_score.replace(']', '')
        idx_score = idx_score.replace(',', '')
        if DEBUG:
          print(idx_score)
        top_indices.append(idx_score.split()[0])
      ret_indices.append(' '.join(top_indices))
    fp.close()
    fp = open(out_file, 'w') 
    fp.write('\n'.join(ret_indices))
    fp.close()

def alignment_to_cluster(ali_file, out_file='cluster.json'):
  def _find_distinct_tokens(data):
    tokens = set()
    for datum in data:
      if 'image_concepts' in datum: 
        tokens = tokens.union(set(datum['image_concepts']))
      elif 'foreign_sent' in datum:
        tokens = tokens.union(set(datum['foreign_sent']))

    return list(tokens)
  
  fp = open(ali_file, 'r')
  align_info_all = json.load(fp)
  fp.close()
         
  classes = _find_distinct_tokens(align_info_all)
  clusters = {c:[] for c in classes}
  for align_info in align_info_all:
    sent = align_info['caption']
    concepts = align_info['image_concepts']
    alignment = align_info['alignment'] 

    if align_info['is_phoneme']:
      sent, alignment = _findPhraseFromPhoneme(sent, alignment)
    
    for w_i, a_i in zip(sent, alignment):
      if a_i >= len(concepts):
        if DEBUG:
          print('alignment index: ', align_info['index'])
          print('a_i out of range: ', a_i, concepts)
        a_i = 0
      clusters[concepts[a_i]].append(w_i)
      clusters[concepts[a_i]] = list(set(clusters[concepts[a_i]]))

  with open(out_file, 'w') as fp:
    json.dump(clusters, fp, indent=4, sort_keys=True)
 
def _findPhraseFromPhoneme(sent, alignment):
  if not hasattr(sent, '__len__') or not hasattr(alignment, '__len__'):
    raise TypeError('sent and alignment should be list')
  if DEBUG:
    print(len(sent), len(alignment))
    print(sent, alignment)
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
  
  return ws, w_align

def resample_alignment(alignment_file, src_feat2wavs_file, trg_feat2wavs_file, out_file):
  with open(alignment_file, "r") as f:
    alignments = json.load(f)

  with open(src_feat2wavs_file, "r") as f:
    src_feat2wavs = json.load(f)
  with open(trg_feat2wavs_file, "r") as f:
    trg_feat2wavs = json.load(f)

  src_ids = sorted(src_feat2wavs, key=lambda x:int(x.split("_")[-1]))
  trg_ids = sorted(trg_feat2wavs, key=lambda x:int(x.split("_")[-1]))

  new_alignments = []  
  for i_ali, ali in enumerate(alignments):
    # TODO: make this faster by making the feat_id convention more consistent
    trg_feat2wav = trg_feat2wavs[trg_ids[i_ali]]
    src_feat2wav = src_feat2wavs[src_ids[i_ali]]
    wavLen = max(src_feat2wav[-1][1], trg_feat2wav[-1][1])
    
    # Frames are automatically assigned to the last frame (convenient if the two feat2wavs have different lengths)
    trg_wav2feat = [-1]*wavLen
    
    alignment = ali["alignment"]

    if DEBUG:
      logging.debug("i_ali: " + str(i_ali))
      logging.debug("src_ids, trg_ids: %s %s" % (src_ids[i_ali], trg_ids[i_ali]))
      logging.debug("# of wav frames in src_feat2wav, # of feat frames: %d %d" % (src_feat2wav[-1][1], len(src_feat2wav))) 
      logging.debug("# pf wav frames in trg_feat2wav, # of feat frames: %d %d" % (trg_feat2wav[-1][1], len(trg_feat2wav)))
      logging.debug("# of frames in alignment: " + str(len(alignment)))
      logging.debug("# of frames in trg_feat2wav: " + str(len(trg_feat2wav)))
        
    for i_seg, seg in enumerate(trg_feat2wav):  
      start = seg[0]
      end = seg[1]
      for t in range(start, end):
        trg_wav2feat[t] = i_seg
    
    new_alignment = [0]*len(trg_feat2wav)
    for i_src in range(len(alignment)):
      for i_wav in range(src_feat2wav[i_src][0], src_feat2wav[i_src][1]):
        if i_wav > len(trg_wav2feat) - 1:
          logging.warning("inconsistent wav lens: %d %d" % (src_feat2wav[-1][1], len(trg_wav2feat)))
          break
        i_trg = trg_wav2feat[i_wav]
        new_alignment[i_trg] = alignment[i_src] 

    new_align_info = deepcopy(ali)
    new_align_info["alignment"] = new_alignment
    new_alignments.append(new_align_info)
  
  with open(out_file, "w") as f:
    json.dump(new_alignments, f, sort_keys=True, indent=4)  

def convert_boundary_to_segmentation(binary_boundary_file, frame_boundary_file):
  binary_boundaries = np.load(binary_boundary_file)
  frame_boundaries = []
  for i, b_vec in enumerate(binary_boundaries):
    print("segmentation %d" % i)
    end_frames = np.nonzero(b_vec)[0]
    
    frame_boundary = [[0, end_frames[0]]]
    for st, end in zip(end_frames[:-1], end_frames[1:]):
      frame_boundary.append([st, end])
    
    if DEBUG:
      print("end_frames: ", end_frames)
      print("frame_boundary: ", frame_boundary) 
    
    frame_boundaries.append(np.asarray(frame_boundary))

  np.save(frame_boundary_file, frame_boundaries) 

def convert_landmark_to_10ms_segmentation(landmark_segment_file, landmarks_file, frame_segment_file):
  lm_segments = np.load(landmark_segment_file)
  lms = np.load(landmarks_file)
  utt_ids = sorted(lms.keys(), key=lambda x:int(x.split('_')[-1]))
  frame_segments = []
  for i, utt_id in enumerate(utt_ids):
    f_seg = []
    for i_lm in lm_segments[i]:
      f_seg.append(lms[utt_id][i_lm])
    frame_segments.append(np.asarray(f_seg))
  np.save(frame_segment_file, frame_segments)

def convert_sec_to_10ms_segmentation(real_time_segment_file, feat2wav_file, frame_segment_file, fs=16000):
  real_time_segments = np.load(real_time_segment_file) 
  with open(feat2wav_file, 'r') as f:
    feat2wavs = json.load(f)
  
  wav2feats = []
  utt_ids = []
  for utt_id, feat2wav in sorted(feat2wavs.items(), key=lambda x:int(x[0].split('_')[-1])):
    print('utt_id: ', utt_id)
    feat_len = len(feat2wav)
    wav_len = feat2wav[-1][1]
    wav2feat = np.zeros((wav_len,), dtype=int)
    for i in range(feat_len):
      wav2feat[feat2wav[i][0]:feat2wav[i][1]] = i
    wav2feats.append(wav2feat)
    utt_ids.append(utt_id)

  frame_segments = {}
  max_gap = [0, 0.]
  max_gap_segment = []
  for i_seg, r_seg in enumerate(real_time_segments.tolist()):
    feat_len = len(feat2wavs[utt_ids[i_seg]])
    wav2feat = wav2feats[i_seg]
    print('utt_id: ', utt_ids[i_seg])
    print('wav_len, wav2feats_len: ', r_seg[-1] * fs, len(wav2feats[i_seg]))

    n_segs = len(r_seg)
    f_seg = [0]
    for i in range(n_segs):
      if wav2feat[int(r_seg[i] * fs)] <= 0 or wav2feat[int(r_seg[i] * fs)] - f_seg[-1] <= 0:
        continue
      else:
        if wav2feat[int(r_seg[i] * fs)] - f_seg[-1] > max_gap[1]:
          max_gap = [i_seg, wav2feat[int(r_seg[i] * fs)] - f_seg[-1]]
          max_gap_segment = [f_seg[-1], wav2feat[int(r_seg[i] * fs)]] 
        if wav2feat[int(r_seg[i] * fs)] >= feat_len - 1:
          continue

        f_seg.append(wav2feat[int(r_seg[i] * fs)] + 1) 
        
    f_seg.append(feat_len)
    frame_segments[utt_ids[i_seg]] = f_seg
  
  print("max_gap: ", max_gap)
  print("max_gap landmark: ", max_gap_segment)
  np.savez(frame_segment_file, **frame_segments)

def convert_txt_to_npy_segment(txt_segment_file, npy_segment_file):
  with open(txt_segment_file, 'r') as f:
    lines = f.read().strip().split('\n\n')    
    segmentations = []
    for seg_txt in lines:
      seg = []
      for line in seg_txt.split('\n'):
        t = float(line.split()[1])
        seg.append(t)
      #print(seg)
      segmentations.append(np.asarray(seg))

    np.save(npy_segment_file, segmentations)

if __name__ == '__main__':
  '''
  postproc = XNMTPostprocessor('../nmt/exp/feb28_phoneme_level_clustering/output/report/')
  postproc.convert_alignment_file('../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json')
  alignment_to_cluster('../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json', '../nmt/exp/feb28_phoneme_level_clustering/output/cluster.json')
  postproc.convert_retrieval_file('../nmt/exp/mar19_phoneme_to_image_norm_over_time/output/phoneme_to_concept.hyp')
  '''
  alignment_file = "../smt/exp/june_24_mfcc_kmeans_mixture=3/flickr30k_pred_alignment.json"
  src_feat2wavs_file = "../data/flickr30k/audio_level/flickr_mfcc_feat2wav.json" 
  ref_feat2wavs_file = "../data/flickr30k/audio_level/flickr30k_gold_alignment.json_feat2wav.json"
  out_file = "../smt/exp/june_24_mfcc_kmeans_mixture=3/pred_alignment_resample.json"
  binary_boundary_file = "../comparison_models/exp/july_20_multimodal_kmeans/boundaries_multimodal-kmeans.npy"
  frame_boundary_file = "../data/flickr30k/audio_level/frame_boundaries_semkmeans.npy"
  txt_syl_segment_file = "../data/flickr30k/audio_level/syllable_boundaries.txt"
  npy_syl_segment_file = "../data/flickr30k/audio_level/syllable_segmentations.npy"
  landmark_file = "flickr_landmarks.npz" #"../data/flickr30k/audio_level/flickr_landmarks.npz"
  feat2wavs_htk_file = "../data/flickr30k/audio_level/flickr_mfcc_cmvn_htk_feat2wav.json"
  #convert_boundary_to_segmentation(binary_boundary_file, frame_boundary_file)
  #resample_alignment(alignment_file, src_feat2wavs_file, ref_feat2wavs_file, out_file)
  #convert_txt_to_npy_segment(txt_syl_segment_file, npy_syl_segment_file)
  convert_sec_to_10ms_segmentation(npy_syl_segment_file, feat2wavs_htk_file, landmark_file) 
