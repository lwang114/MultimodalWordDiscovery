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

def preemphasis(signal, coeff=0.97):
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])

if __name__ == "__main__":
  data_dir = "../data/flickr30k/audio_level/"
  audio_info_file = data_dir + "flickr30k_gold_alignment.json"
  audio_dir = "/home/lwang114/data/flickr_audio/"
  utt2spk_file = audio_dir + "wav2spk.txt"
  feat_configs = {} 
  out_dir = data_dir
  feat_extractor = FlickrFeaturePreprocessor(audio_info_file, audio_dir, utt2spk_file)
  feat_extractor.extractMFCC(feat_configs, out_dir)
