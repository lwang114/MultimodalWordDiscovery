# Driver code to run acoustic word discovery systems
# by Kamper, Livescu and Goldwater 2016
import numpy as np
from segmentalist.segmentalist.unigram_acoustic_wordseg import *
from segmentalist.segmentalist.kmeans_acoustic_wordseg import * 
from bucktsong_segmentalist.downsample.downsample import *   
import segmentalist.segmentalist.fbgmm
import segmentalist.segmentalist.gaussian_components_fixedvar
import segmentalist.segmentalist.kmeans

#import argparse
#import time
#import logging
#from scipy import signal

logger = logging.getLogger(__name__)
i_debug_monitor = 0  # 466  # the index of an utterance which is to be monitored
segment_debug_only = False  # only sample the debug utterance
DEBUG = False

# TODO: Make a wrapper class
def downsample(y, n, args):
  # Downsample
  if args.technique == "interpolate":
      x = np.arange(y.shape[1])
      f = interpolate.interp1d(x, y, kind="linear")
      x_new = np.linspace(0, y.shape[1] - 1, n)
      y_new = f(x_new).flatten(flatten_order) #.flatten("F")
  elif args.technique == "resample":
      
      y_new = signal.resample(y.astype("float32"), n, axis=1).flatten(flatten_order) #.flatten("F")
  elif args.technique == "rasanen":
      # Taken from Rasenen et al., Interspeech, 2015
      d_frame = y.shape[0]
      n_frames_in_multiple = int(np.floor(y.shape[1] / n)) * n
      y_new = np.mean(
          y[:, :n_frames_in_multiple].reshape((d_frame, n, -1)), axis=-1
          ).flatten(flatten_order) #.flatten("F")
  return y_new

parser = argparse.ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=300, help="Dimension of the embedding vector")
parser.add_argument("--n_slices_min", type=int, default=10, help="Minimum length of the segment")
parser.add_argument("--n_slices_max", type=int, default=40, help="Maximum length of the segment")
parser.add_argument("--technique", choices={"resample", "interpolate", "rasanen"}, default="resample", help="Downsampling technique")
parser.add_argument("--am_class", choices={"fbgmm", "kmeans"}, help="Class of acoustic model")
parser.add_argument("--am_K", type=int, default=10, help="Number of clusters")
parser.add_argument("--exp_dir", type=str, default='./', help="Experimental directory")
args = parser.parse_args()
print(args)

datapath = "../data/flickr30k/audio_level/flickr_bnf_subset.npz"
# Generate acoustic embeddings, vec_ids_dict and durations_dict 
audio_feats = np.load(datapath)
embedding_mats = {}
vec_ids_dict = {}
durations_dict = {}
landmarks_dict = {}

start_step = 0
if start_step == 0:
  print("Start extracting acoustic embeddings")
  begin_time = time.time()
  for feat_id in sorted(audio_feats.keys(), key=lambda x:int(x.split('_')[-1])):
    #print (feat_id)
    feat_mat = audio_feats[feat_id]
    n_slices = feat_mat.shape[0]
    feat_dim = feat_mat.shape[1]
    assert args.embed_dim % feat_dim == 0
    assert args.n_slices_min >= args.embed_dim / feat_dim
    embed_mat = np.zeros((n_slices * (1 + n_slices) / 2, args.embed_dim))
    vec_ids = -1 * np.ones((n_slices * (1 + n_slices) / 2,))
    durations = np.nan * np.ones((n_slices * (1 + n_slices) / 2,))

    i_embed = 0        
    # Store the vec_ids using the mapping i_embed = end * (end - 1) / 2 + start (following unigram_acoustic_wordseg.py)
    for cur_start in range(n_slices):
        for cur_end in range(cur_start + args.n_slices_min, min(n_slices, cur_start + args.n_slices_max)):
            cur_end += 1
            t = cur_end
            
            i = t*(t - 1)/2
            vec_ids[i + cur_start] = i_embed
            #print cur_start, cur_end, i + cur_start, i_embed
            n_down_slices = args.embed_dim / feat_dim
            embed_mat[i_embed] = downsample(feat_mat[cur_start:cur_end].T, n_down_slices, args)
             
            durations[i + cur_start] = cur_end - cur_start
            i_embed += 1 


    vec_ids_dict[feat_id] = vec_ids
    embedding_mats[feat_id] = embed_mat
    durations_dict[feat_id] = durations 
    landmarks_dict[feat_id] = np.arange(n_slices).tolist()
  print("Take %0.5f s to finish extracting embedding vectors !" % (time.time()-begin_time))
  #np.savez("embedding_mats.npz", **embedding_mats)
  np.savez("vec_ids_dict.npz", **vec_ids_dict)
  np.savez("durations_dict.npz", **durations_dict)
  np.savez("landmarks_dict.npz", **landmarks_dict)  

if start_step <= 1:
  print("Start training segmentation models")
  begin_time = time.time()
  #embedding_mats = np.load("embedding_mats.npz")
  #vec_ids_dict = np.load("vec_ids_dict.npz")
  #durations_dict = np.load("durations_dict.npz")
  #landmarks_dict = np.load("landmarks_dict.npz")

  # Acoustic model parameters
  segmenter = None
  if args.am_class == "fbgmm":
    am_class = fbgmm.FBGMM
    am_alpha = 10.
    am_K = 1000
    m_0 = np.zeros(D)
    k_0 = 0.05
    # S_0 = 0.025*np.ones(D)
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
    segmenter = UnigramAcousticWordseg(
      am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict,
      durations_dict, landmarks_dict=landmarks_dict, p_boundary_init=0.5, beta_sent_boundary=-1, n_slices_max=args.n_slices_max
      )
  
  elif args.am_class == "kmeans":
    am_K = args.am_K
    # Initialize model
    segmenter = SegmentalKMeansWordseg(am_K, embedding_mats, vec_ids_dict,
      durations_dict, landmarks_dict=landmarks_dict, p_boundary_init=0.5, n_slices_max=args.n_slices_max
      )
  
  # Perform sampling
  if args.am_class == "fbgmm":
    record = segmenter.gibbs_sample(3)
    #sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
  else:
    record = segmenter.segment(3, 3)
    sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
    print("Take %0.5f s to finish training !" % (time.time() - begin_time))
  #np.save("%ssum_neg_len_sqrd_norm_%s.npy" % (args.exp_dir, args.am_class), sum_neg_len_sqrd_norm)
  np.save("%sboundaries_%s.npy" % (args.exp_dir, args.am_class), segmenter.utterances.boundaries)

if start_step <= 2:
  boundaries = np.load("%sboundaries_%s.npy" % (args.exp_dir, args.am_class))
  #for i, feat_id in enumerate(sorted(audio_feats.keys(), key=lambda x:int(x.split('_')[-1]))):
  #print boundaries[i]

  # Evaluate (full-coverage boundary scores, concept-level boundary scores, precision for head concept classification)
