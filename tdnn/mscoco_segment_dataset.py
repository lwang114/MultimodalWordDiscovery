import torch
import librosa
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

EPS = 1e-9
# This function is from DAVEnet (https://github.com/dharwath/DAVEnet-pytorch)
def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class MSCOCOSegmentDataset(Dataset):
  def __init__(self, audio_root_path, segment_file, phone2idx_file, feat_configs=None):
    self.n_mfcc = feat_configs.get('n_mfcc', 40)
    self.order = feat_configs.get('order', 2)
    self.coeff = feat_configs.get('coeff', 0.97)
    self.dct_type = feat_configs.get('dct_type', 3)
    self.skip_ms = feat_configs.get('skip_size', 10)
    self.window_ms = feat_configs.get('window_len', 25)
    compute_cmvn = feat_configs.get('compute_cmvn', False)
    self.phone_keys = []
    self.phone_labels = []
    self.segmentations = []
    self.audio_root_path = audio_root_path
    with open(segment_file, 'r') as f:
      i = 0
      for line in f:
        # XXX
        # if i > 100:
        #   break
        # i += 1 

        k, phn, start, end = line.strip().split()
        self.phone_labels.append(phn)
        self.phone_keys.append(k)
        self.segmentations.append([start, end])

    with open(phone2idx_file, 'r') as f:
      self.phone2idx = json.load(f)

  def __len__(self):
    return len(self.phone_keys)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    sr, y = io.wavfile.read(self.audio_root_path + self.phone_keys[idx] + '.wav')

    start_ms, end_ms = self.segmentations[idx]
    start, end = int(float(start_ms) * sr / 1000), int(float(end_ms) * sr / 1000) 
    n_fft = int(self.window_ms * sr / 1000)
    hop_length = int(self.skip_ms * sr / 1000)
    if end <= start: 
      print('empty segment')
      end = start + 1

    seg = y[start:end]
    seg = preemphasis(seg, self.coeff) 
    # y = preemphasis(y, self.coeff)
    # N = y.shape[0]
    # mfcc = librosa.feature.mfcc(seg, sr=sr, n_mfcc=self.n_mfcc, dct_type=self.dct_type, n_fft=n_fft, hop_length=hop_length)    
    # n_frames_mfcc = mfcc.shape[1]
    # mfcc = mfcc[:, int(start / hop_length):int(end / hop_length)]
    mfcc = librosa.feature.mfcc(seg, sr=sr, n_mfcc=self.n_mfcc, dct_type=self.dct_type, n_fft=n_fft, hop_length=hop_length)
    mfcc -= np.mean(mfcc)
    mfcc /= max(np.sqrt(np.var(mfcc)), EPS)
    mfcc = self.convert_to_fixed_length(mfcc)
    label = self.phone2idx[self.phone_labels[idx]] 
    # TODO
    # if self.compute_cmvn:
    return torch.FloatTensor(mfcc), label

  def convert_to_fixed_length(self, mfcc, N=15):
    T = mfcc.shape[1] 
    gap = abs(N - T)
    l =  int(gap / 2)
    r = gap - l
    if T < N:
      mfcc = np.pad(mfcc, ((0, 0), (l, r)), 'constant', constant_values=(0))
    elif T > N:
      mfcc = mfcc[:, l:-r]
    return mfcc  
