import numpy as np
import librosa
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

    with open(utt2spk_file, 'r') as f:
      lines = f.read().split("\n")
      self.utt2spk = {fn:spk for line in lines for fn, spk in line}
      self.spk_count = {spk:0 for spk in self.utt2spk}
    
    self.spk_means = {spk: np.zeros((feat_dim,)) for spk in self.spk_count}
    self.spk_vars = {spk: np.zeros((feat_dim,)) for spk in self.spk_count}

  def extractMFCC(self, feat_configs, out_dir):
    n_mfcc = feat_configs.get("n_mfcc", 13)
    order = feat_configs.get("order", 2)
    coeff = feat_configs.get("coeff", 0.97)
    dct_type = feat_configs.get("dct_type", 3)
    compute_cmvn = feat_configs.get("compute_cmvn", True)

    for i, audio_info_i in enumerate(self.audio_info):
      audio_id = audio_info_i["audio_id"]
      img_id = audio_info_i["img_id"]
      
      for fn in self.audio_files:
        cur_img_id, _, cur_audio_id = fn.split(".")[0].split("_") 
        if cur_img_id == img_id and cur_audio_id == audio_id:
          break

      feat_id = '_'.join([fn.split('.')[0], str(i)])

      sr, y = io.wavfiles.read(self.audio_dir + fn) 
      y = preemphasis(y, coeff)
      N = y.shape[1]

      mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type)
      mfcc2wav = []
      N_mfcc = mfcc.shape[1]
      n = int(N / N_mfcc)
      for i_mf in range(N_mfcc):
        if i_mf == N_mfcc - 1:
          mfcc2wav[i_mf] = range(i_mf * n, N)
        else:
          mfcc2wav[i_mf] = range(i_mf * n, i_mf * (n + 1))
      
      if order >= 1:
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc = np.concatenate([mfcc, mfcc_delta], axis=0)
      if order >= 2:
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, mfcc_delta2], axis=0)
      
      self.mfccs[feat_id] = mfcc.T
      self.mfcc2wavs[feat_id] = mfcc2wav
      self.spk_count[self.utt2spk[fn]] += 1
    
    np.savez(out_dir + "flickr_mfcc.npz", **mfcc)
    with open(out_dir + "flickr_mfcc_feat2wav.json", "w") as f:
      json.dump(mfcc2wavs, f)

    if compute_cmvn:
      self.cmvn(feat_configs, out_dir)
    
    np.savez(out_dir + "flickr_mfcc.npz", **mfcc) 

  def cmvn(self, feat_configs, out_dir):
    n_mfcc = feat_configs.get("n_mfcc", 13)
    order = feat_configs.get("order", 2)
    
    feat_dim = n_mfcc * (order + 1)

    for feat_id in sorted(self.mfccs, key=lambda x:x.split('_')[-1]):
      utt_id = "_".join(feat_id.split('_')[:-1]) + ".wav"
      spk_id = self.utt2spk[utt_id]
      self.spk_means[spk_id] += 1 / self.spk_count[spk_id] * np.sum(self.mfccs[feat_id], axis=0)
      
    for feat_id in sorted(self.mfccs, key=lambda x:x.split('_')[-1]):
      utt_id = "_".join(feat_id.split('_')[:-1]) + ".wav"
      spk_id = self.utt2spk[utt_id]
      self.spk_vars[spk_id] += 1 / self.spk_count[spk_id] * np.sum((self.mfccs[feat_id] - self.spk_means[spk_id]) ** 2, axis=0)

    np.savez(out_dir+"flickr_mfcc_spk_means.npz")
    np.savez(out_dir+"flickr_mfcc_spk_variance.npz")
    
    for feat_id in sorted(self.mfccs, key=lambda x:x.split('_')[-1]):
      utt_id = "_".join(feat_id.split('_')[:-1]) + ".wav"
      spk_id = self.utt2spk[utt_id]
      self.mfccs[feat_id] = (self.mfccs[feat_id] - self.spk_means[spk_id]) / np.sqrt(self.spk_vars) 

def preemphasis(signal, coeff=0.97):
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])

if __name__ == "__main__":
  audio_info_file =
  audio_dir = 
  utt2spk_file = ""
  feat_configs = {} 
  out_dir = ""
  feat_extractor = FlickrFeaturePreprocessor(audio_info_file, audio_dir, utt2spk_file)
  feat_extractor.extractMFCC(feat_configs, out_dir)
