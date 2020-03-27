import torch
import librosa
import scipy.io as io
from sphfile import SPHFile
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

class AudioSequenceDataset(Dataset):
  def __init__(self, audio_root_path, audio_sequence_file, phone2idx_file, feat_configs=None):
    # Inputs:
    # ------
    #     audio_root_path: str containing the root to the audio files
    #     audio_sequence_file: name of the meta info file. Each row contains the string of audio filename and phone sequence of the audio
    #     phone2idx_file: name of the json file containing the mapping between each phone symbol and an integer id
    
    self.n_mfcc = feat_configs.get('n_mfcc', 40)
    # self.order = feat_configs.get('order', 2)
    self.coeff = feat_configs.get('coeff', 0.97)
    self.dct_type = feat_configs.get('dct_type', 3)
    self.skip_ms = feat_configs.get('skip_size', 10)
    self.window_ms = feat_configs.get('window_len', 25)
    compute_cmvn = feat_configs.get('compute_cmvn', False)
    self.audio_filenames = []
    self.phone_sequences = []
    self.ES = -1
    self.max_nphones = feat_configs.get('max_num_phones', 100)
    self.max_nframes = feat_configs.get('max_num_frames', 1000)
    # XXX
    self.max_nphones = 100
    self.max_nframes = 1000

    self.audio_root_path = audio_root_path
    with open(audio_sequence_file, 'r') as f:
      i = 0
      for line in f:
        # XXX
        # if i > 16:
        #   break
        i += 1 
        audio_info = line.strip().split()        
        self.phone_sequences.append(audio_info[1:])
        self.audio_filenames.append(audio_info[0])

    with open(phone2idx_file, 'r') as f:
      self.phone2idx = json.load(f)

  def __len__(self):
    return len(self.audio_filenames)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    audio_filename = self.audio_filenames[idx]
    try:
      sr, y = io.wavfile.read(self.audio_root_path + audio_filename)
    except:
      if audio_filename.split('.')[-1] == 'wav':
        audio_filename_sph = '.'.join(audio_filename.split('.')[:-1]+['WAV'])
        sph = SPHFile(self.audio_root_path + audio_filename_sph)
        sph.write_wav(self.audio_root_path + audio_filename)
      sr, y = io.wavfile.read(self.audio_root_path + audio_filename)

    y = preemphasis(y, self.coeff) 

    n_fft = int(self.window_ms * sr / 1000)
    hop_length = int(self.skip_ms * sr / 1000)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=self.n_mfcc, dct_type=self.dct_type, n_fft=n_fft, hop_length=hop_length)
    mfcc -= np.mean(mfcc)
    mfcc /= max(np.sqrt(np.var(mfcc)), EPS)
    nframes = min(mfcc.shape[1], self.max_nframes)
    mfcc = self.convert_to_fixed_length(mfcc)
    mfcc = mfcc.T

    phone_seq = self.phone_sequences[idx]
    nphones = min(len(phone_seq), self.max_nphones)
    labels = [self.phone2idx[phone_seq[i]] if i < len(phone_seq) else self.ES for i in range(self.max_nphones)] 
    # labels = [self.phone2idx[phone_seq[i]] for i in range(min(self.max_nphones, len(phone_seq)))]
    # TODO 
    # if self.compute_cmvn:
    return torch.FloatTensor(mfcc), torch.IntTensor(labels), nframes, nphones

  def convert_to_fixed_length(self, mfcc):
    T = mfcc.shape[1] 
    pad = abs(self.max_nframes - T)
    if T < self.max_nframes:
      mfcc = np.pad(mfcc, ((0, 0), (0, pad)), 'constant', constant_values=(0))
    elif T > self.max_nframes:
      mfcc = mfcc[:, :-pad]
    return mfcc  
