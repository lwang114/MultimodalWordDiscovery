import numpy as np
import librosa
import json
import os
import scipy.io as io

class FlickrFeaturePreprocessor:
  def __init__(self, audio_info_file, audio_dir, utt2spk_file):
    with open(audio_info_file, 'r') as f:
      self.audio_info = json.load(f)
    self.audio_dir = audio_dir
    self.audio_files = os.listdir(self.audio_dir+'wavs/')
    self.mfccs = {}
    self.mfcc2wavs = {} 
    self.spk_means = {}
    self.spk_vars = {}

    with open(utt2spk_file, 'r') as f:
      lines = f.read().strip().split("\n")
      self.utt2spk = {line.split()[0]:line.split()[1] for line in lines}
      self.spk_count = {spk:0 for spk in set(self.utt2spk.values())}
         
  def extractMFCC(self, feat_configs, out_dir):
    n_mfcc = feat_configs.get("n_mfcc", 13)
    order = feat_configs.get("order", 2)
    coeff = feat_configs.get("coeff", 0.97)
    dct_type = feat_configs.get("dct_type", 3)
    compute_cmvn = feat_configs.get("compute_cmvn", True)

    for i, audio_info_i in enumerate(self.audio_info):
      audio_id = audio_info_i["capt_id"].split("_")[-1]
      img_id = audio_info_i["image_id"]
      print(audio_id, img_id)
      
      found = False
      for fn in self.audio_files:
        cur_img_id, _, cur_audio_id = fn.split(".")[0].split("_") 
        if cur_img_id == img_id and cur_audio_id == audio_id:
          found = True
          break
      if not found:
        print("image id not found, check the filename format")
        assert False

      feat_id = '_'.join([fn.split('.')[0], str(i)])

      sr, y = io.wavfile.read(self.audio_dir + 'wavs/' + fn) 
     
      y = preemphasis(y, coeff)
      N = y.shape[0]

      mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type)
      n_frames_mfcc = mfcc.shape[1]
      mfcc2wav = []
      
      n = int(N / n_frames_mfcc)
      for i_mf in range(n_frames_mfcc):
        if i_mf == n_frames_mfcc - 1:
          mfcc2wav.append((i_mf * n, N))
        else:
          mfcc2wav.append((i_mf * n, (i_mf + 1) * n))
      
      if order >= 1:
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc = np.concatenate([mfcc, mfcc_delta], axis=0)
      if order >= 2:
        mfcc_delta2 = librosa.feature.delta(mfcc[:n_mfcc], order=2)
        mfcc = np.concatenate([mfcc, mfcc_delta2], axis=0)
     
      self.mfccs[feat_id] = mfcc.T
      self.mfcc2wavs[i] = mfcc2wav
      self.spk_count[self.utt2spk[fn]] += mfcc.shape[1]
      
    np.savez(out_dir + "flickr_mfcc.npz", **self.mfccs)
    
    with open(out_dir + "flickr_mfcc_feat2wav.json", "w") as f:
      json.dump(self.mfcc2wavs, f, indent=4, sort_keys=True)
    
    if compute_cmvn:
      self.cmvn(feat_configs, out_dir)
    
    np.savez(out_dir + "flickr_mfcc_cmvn.npz", **self.mfccs)     

  def cmvn(self, feat_configs, out_dir):
    n_mfcc = feat_configs.get("n_mfcc", 13)
    order = feat_configs.get("order", 2)
    
    feat_dim = n_mfcc * (order + 1)
    self.spk_means = {spk: np.zeros((feat_dim,)) for spk in self.spk_count}
    self.spk_vars = {spk: np.zeros((feat_dim,)) for spk in self.spk_count}

    for feat_id in sorted(self.mfccs, key=lambda x:x.split('_')[-1]):
      utt_id = "_".join(feat_id.split('_')[:-1]) + ".wav"
      spk_id = self.utt2spk[utt_id]
      self.spk_means[spk_id] += 1. / self.spk_count[spk_id] * np.sum(self.mfccs[feat_id], axis=0)
      
    for feat_id in sorted(self.mfccs, key=lambda x:x.split('_')[-1]):
      utt_id = "_".join(feat_id.split('_')[:-1]) + ".wav"
      spk_id = self.utt2spk[utt_id]
      self.spk_vars[spk_id] += 1. / self.spk_count[spk_id] * np.sum((self.mfccs[feat_id] - self.spk_means[spk_id]) ** 2, axis=0)

    np.savez(out_dir+"flickr_mfcc_spk_means.npz")
    np.savez(out_dir+"flickr_mfcc_spk_variance.npz")
    
    for feat_id in sorted(self.mfccs, key=lambda x:x.split('_')[-1]):
      utt_id = "_".join(feat_id.split('_')[:-1]) + ".wav"
      spk_id = self.utt2spk[utt_id]
      if (self.spk_vars[spk_id] == 0).any():
        print(spk_id)
      self.mfccs[feat_id] = (self.mfccs[feat_id] - self.spk_means[spk_id]) / np.sqrt(self.spk_vars[spk_id]) 
  
  def extract_concept_word_kamper_embeddings(self, feat_file, alignment_file, segmentation_file, embed_dim, file_prefix='flickr_concept_kamper_embeddings', save_segmentation=True):
    feat_npz = np.load(feat_file)
    segment_npz = np.load(segmentation_file)
    segment_ids = [k for k in sorted(segment_npz, key=lambda x:int(x.split('_')[-1]))]
    # XXX
    with open(alignment_file, 'r') as f:
      align_info = json.load(f)
    align_info = align_info 

    concept_word_segments = {}
    concept_word_embeddings = {} 
    for i, seg_id in enumerate(segment_ids):
      print(seg_id)
      feats = feat_npz[seg_id]
      segs = segment_npz[seg_id]
      aligns = align_info[i]['alignment'] 
      concept_segs = []
      concept_feats = []
      for start, end in zip(segs[:-1], segs[1:]):
        if len(set(aligns[start:end])) > 1:
          print('non-uniform alignment within segment: ', start, end, aligns[start:end])
        
        if np.sum(np.asarray(aligns[start:end]) > 0) > 1./2 * (end-start):
          concept_segs.append([start, end])
          concept_feats.append(feats[start:end])
      concept_word_segments[seg_id] = concept_segs
      concept_embeds = self.get_concept_embeds(concept_feats, embed_dim)
      concept_word_embeddings[seg_id] = concept_embeds
    np.savez(file_prefix+'.npz', **concept_word_embeddings)
    if save_segmentation:
      np.savez(file_prefix+'_segmentation.npz', **concept_word_segments) 

  def get_concept_embeds(self, x, embed_dim, frame_dim=12):
    embeddings = []
    for seg in x:
      #print("seg.shape", seg.shape)
      #print("seg:", segmentation[i_w+1])
      #print("embed of seg:", self.embed(seg))
      embeddings.append(embed(seg, embed_dim, frame_dim=frame_dim))  
    return np.array(embeddings)
     
  def convertMatToNpz(feat_mat_file, feat_npz_file, utterance_ids_file=None, feat2wav_file=None):
    feat_mat = io.loadmat(feat_mat_file)['F']
    n_feats = len(feat_mat)
    if utterance_ids_file is None:
      utterance_ids = [str(i) for i in range(n_feats)]  
    else:
      with open(utterance_ids_file, 'r') as f:
        utterance_ids = json.load(f)

    for utt_id, feat in zip(utterance_ids, feat_mat):
      feat_dict[utt_id] = feat
    np.savez(feat_npz_file, **feat_dict)
    
    if feat2wav_file:
      feat2wavs = {}
      for i, (audio_info_i, feat) in enumerate(zip(self.audio_info, feat_mat)):
        audio_id = audio_info_i["capt_id"].split("_")[-1]
        img_id = audio_info_i["image_id"]
      
        found = False
        for fn in self.audio_files:
          cur_img_id, _, cur_audio_id = fn.split(".")[0].split("_") 
          if cur_img_id == img_id and cur_audio_id == audio_id:
            found = True
            break
        if not found:
          print("image id not found, check the filename format")
          assert False

        sr, y = io.wavfile.read(self.audio_dir + 'wavs/' + fn) 
        N = y.shape[0]

        feat_len = feat.shape[0]
        n = int(N / feat_len)
        print(audio_id, img_id, n)
        feat2wav = []
        for i_f in range(feat_len):
          if i_f == feat_len - 1:
            feat2wav.append((i_f * n, N))
          else:
            feat2wav.append((i_f * n, (i_f + 1) * n))
        feat2wavs[i] = feat2wav
      
      with open(feat2wav_file, 'w') as f:
        json.dump(feat2wavs, f)

class MSCOCOAudioFeaturePreprocessor:
  def __init__(self, audio_info_file, audio_dir):
    with open(audio_info_file, 'r') as f:
      audio_info = json.load(f)
      self.audio_info = [audio_info[k] for k in sorted(audio_info, key=lambda x:int(x.split('_')[-1]))]
    self.audio_dir = audio_dir
    self.mfccs = {}
    self.mfcc2wavs = {} 
    self.spk_means = {}
    self.spk_vars = {}
    self.utt2spk = {}
    self.spk_counts = {}

  def extractMFCC(self, feat_configs, out_dir):
    n_mfcc = feat_configs.get("n_mfcc", 14)
    order = feat_configs.get("order", 2)
    coeff = feat_configs.get("coeff", 0.97)
    dct_type = feat_configs.get("dct_type", 3)
    compute_cmvn = feat_configs.get("compute_cmvn", True)

    # XXX
    for i, audio_info_i in enumerate(self.audio_info):
      index = 'arr_'+str(i)
      data_ids = audio_info_i["data_ids"]
      print(data_ids)
      self.mfccs[index] = []
      for data_id in data_ids:
        audio_id = '_'.join(data_id[1].split('_')[:-1])
        print(audio_id)
        sr, y = io.wavfile.read(self.audio_dir + 'wav/' + audio_id+'.wav') 
        start_ms, end_ms = data_id[2], data_id[3]
        spk = data_id[-1]
        start = int(start_ms * sr / 1000.)
        end = int(end_ms * sr / 1000.)

        y = preemphasis(y, coeff)
        y = y[start:end]
        mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type)
        if order >= 1:
          mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
          mfcc = np.concatenate([mfcc, mfcc_delta], axis=0)
        if order >= 2:
          mfcc_delta2 = librosa.feature.delta(mfcc[:n_mfcc], order=2, mode='nearest')
          mfcc = np.concatenate([mfcc, mfcc_delta2], axis=0)

        n_frames_mfcc = mfcc.shape[1]
        
        self.mfccs[index].append(mfcc.T)
        if spk in self.spk_counts:
          self.spk_counts[spk] += 1
        else:
          self.spk_counts[spk] = 1

        if index in self.utt2spk:
          self.utt2spk[index].append(spk)
        else: 
          self.utt2spk[index] = [spk]
    np.savez(out_dir + "mscoco_mfcc.npz", **self.mfccs)
    
    if compute_cmvn:
      self.cmvn(feat_configs, out_dir)
    
    np.savez(out_dir + "mscoco_mfcc_cmvn.npz", **self.mfccs)     

  def cmvn(self, feat_configs, out_dir):
    n_mfcc = feat_configs.get("n_mfcc", 14)
    order = feat_configs.get("order", 2)
    
    feat_dim = n_mfcc * (order + 1)
    self.spk_means = {spk: np.zeros((feat_dim,)) for spk in self.spk_counts}
    self.spk_vars = {spk: np.zeros((feat_dim,)) for spk in self.spk_counts}

    # XXX
    for feat_id in sorted(self.mfccs, key=lambda x:int(x.split('_')[-1])):
      spk_ids = self.utt2spk[feat_id]
      for afeat, spk_id in zip(self.mfccs[feat_id], spk_ids):
        self.spk_means[spk_id] += 1. / self.spk_counts[spk_id] * np.sum(afeat, axis=0)
      
    for feat_id in sorted(self.mfccs, key=lambda x:int(x.split('_')[-1])):
      spk_ids = self.utt2spk[feat_id]
      for afeat, spk_id in zip(self.mfccs[feat_id], spk_ids):
        self.spk_vars[spk_id] += 1. / self.spk_counts[spk_id] * np.sum((afeat - self.spk_means[spk_id]) ** 2, axis=0)

    np.savez(out_dir+"mscoco_mfcc_spk_means.npz")
    np.savez(out_dir+"mscoco_mfcc_spk_variance.npz")
    
    for feat_id in sorted(self.mfccs, key=lambda x:int(x.split('_')[-1])):
      spk_ids = self.utt2spk[feat_id]
      for i, (afeat, spk_id) in enumerate(zip(self.mfccs[feat_id], spk_ids)):
        self.mfccs[feat_id][i] = (afeat - self.spk_means[spk_id]) / np.sqrt(self.spk_vars[spk_id]) 

def preemphasis(signal, coeff=0.97):
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def embed(y, embed_dim, frame_dim=None, technique="resample"):
  #assert embed_dim % self.audio_feat_dim == 0
  if frame_dim: 
    y = y[:, :frame_dim].T
  else:
    y = y.T
    frame_dim = y.shape[-1]

  n = int(embed_dim / frame_dim)
  if y.shape[0] == 1: 
    y_new = np.repeat(y, n)   

  #if y.shape[0] <= n:
  #  technique = "interpolate" 
       
  #print(embed_dim / frame_dim)
  if technique == "interpolate":
      x = np.arange(y.shape[1])
      f = interpolate.interp1d(x, y, kind="linear")
      x_new = np.linspace(0, y.shape[1] - 1, n)
      y_new = f(x_new).flatten(ORDER) #.flatten("F")
  elif technique == "resample":
      y_new = signal.resample(y.astype("float32"), n, axis=1).flatten(ORDER) #.flatten("F")
  elif technique == "rasanen":
      # Taken from Rasenen et al., Interspeech, 2015
      n_frames_in_multiple = int(np.floor(y.shape[1] / n)) * n
      y_new = np.mean(
          y[:, :n_frames_in_multiple].reshape((d_frame, n, -1)), axis=-1
          ).flatten(ORDER) #.flatten("F")
  return y_new
 

if __name__ == "__main__":
  #data_dir = "../data/flickr30k/audio_level/"
  #audio_info_file = data_dir + "flickr30k_gold_alignment.json"
  #audio_dir = "/home/lwang114/data/flickr_audio/"#"/ws/ifp-53_2/hasegawa/lwang114/data/flickr_audio/"
  #utt2spk_file = audio_dir + "wav2spk.txt"
  #feat_mat_file = data_dir + "flickr_embeddings.mat" #"flickr_mfcc_cmvn_htk.mat"
  #feat_npz_file = data_dir + "flickr_embeddings.npz" #"flickr_mfcc_cmvn_htk.npz"
  #utterance_ids_file = data_dir + "ids_to_utterance_labels.json"
  
  feat_configs = {} 
  #out_dir = data_dir
  #feat_extractor = FlickrFeaturePreprocessor(audio_info_file, audio_dir, utt2spk_file)
  out_dir = '.'
  audio_dir = '/home/lwang114/data/mscoco/val2014/'
  audio_info_file = '../data/mscoco/mscoco_subset_concept_info_power_law.json' 
  feat_extractor = MSCOCOAudioFeaturePreprocessor(audio_info_file, audio_dir)
  feat_extractor.extractMFCC(feat_configs, out_dir)

  #feat_extractor.convertMatToNpz(feat_mat_file, feat_npz_file, utterance_ids_file)
