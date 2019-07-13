# Driver code to run acoustic word discovery systems
# by Kamper, Livescu and Goldwater 2016
import numpy as np
import json
from segmentalist.segmentalist.unigram_acoustic_wordseg import *
from segmentalist.segmentalist.kmeans_acoustic_wordseg import * 
from bucktsong_segmentalist.downsample.downsample import *   
import segmentalist.segmentalist.fbgmm as fbgmm
import segmentalist.segmentalist.gaussian_components_fixedvar as gaussian_components_fixedvar
import segmentalist.segmentalist.kmeans as kmeans

#import argparse
#import time
#import logging
#from scipy import signal

logger = logging.getLogger(__name__)
i_debug_monitor = 0  # 466  # the index of an utterance which is to be monitored
segment_debug_only = False  # only sample the debug utterance
DEBUG = True

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
parser.add_argument("--embed_dim", type=int, default=180, help="Dimension of the embedding vector")
parser.add_argument("--n_slices_min", type=int, default=10, help="Minimum length of the segment")
parser.add_argument("--n_slices_max", type=int, default=40, help="Maximum length of the segment")
parser.add_argument("--technique", choices={"resample", "interpolate", "rasanen"}, default="resample", help="Downsampling technique")
parser.add_argument("--am_class", choices={"fbgmm", "kmeans"}, help="Class of acoustic model")
parser.add_argument("--am_K", type=int, default=10, help="Number of clusters")
parser.add_argument("--exp_dir", type=str, default='./', help="Experimental directory")
parser.add_argument("--feat_type", type=str, choices={"mfcc", "bn"}, help="Acoustic feature type")
args = parser.parse_args()
print(args)
if args.feat_type == "bn":
  datapath = "../data/flickr30k/audio_level/flickr_bnf_all_src.npz"
else:
  datapath = "../data/flickr30k/audio_level/flickr_mfcc_cmvn.npz"

# Generate acoustic embeddings, vec_ids_dict and durations_dict 
audio_feats = np.load(datapath)
embedding_mats = {}
vec_ids_dict = {}
durations_dict = {}
landmarks_dict = {}

start_step = 3
if start_step == 0:
  print("Start extracting acoustic embeddings")
  begin_time = time.time()
  for feat_id in sorted(audio_feats.keys(), key=lambda x:int(x.split('_')[-1])):
    print (feat_id)
    feat_mat = audio_feats[feat_id]
    if feat_id.split("_")[0] == "3652859271":
      feat_mat = feat_mat[:1000]
    n_slices = feat_mat.shape[0]
    feat_dim = feat_mat.shape[1]
    assert args.embed_dim % feat_dim == 0
    assert args.n_slices_min >= args.embed_dim / feat_dim
    embed_mat = np.zeros(((args.n_slices_max - args.n_slices_min + 1)*n_slices, args.embed_dim))
    vec_ids = -1 * np.ones((n_slices * (1 + n_slices) / 2,))
    durations = np.nan * np.ones((n_slices * (1 + n_slices) / 2,))

    i_embed = 0        
    # Store the vec_ids using the mapping i_embed = end * (end - 1) / 2 + start (following unigram_acoustic_wordseg.py)
    for cur_start in range(n_slices):
        for cur_end in range(cur_start + args.n_slices_min - 1, min(n_slices, cur_start + args.n_slices_max)):
    
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
  np.savez(args.exp_dir+"embedding_mats.npz", **embedding_mats)
  np.savez(args.exp_dir+"vec_ids_dict.npz", **vec_ids_dict)
  np.savez(args.exp_dir+"durations_dict.npz", **durations_dict)
  np.savez(args.exp_dir+"landmarks_dict.npz", **landmarks_dict)  

if start_step <= 1:
  print("Start cleaning up dataset ...")
  begin_time = time.time()
  '''embedding_mats = dict(np.load(args.exp_dir+"embedding_mats.npz"))
  vec_ids_dict = dict(np.load(args.exp_dir+"vec_ids_dict.npz"))
  durations_dict = dict(np.load(args.exp_dir+"durations_dict.npz"))
  landmarks_dict = dict(np.load(args.exp_dir+"landmarks_dict.npz"))

  for embed_id in embedding_mats:
    if embedding_mats[embed_id].shape[0] > 500500:
      logging.debug("sentence with id %s is exceedingly long" % embed_id)
    
    if embed_id.split("_")[0] == "3652859271":
      embedding_mats[embed_id] = embedding_mats[embed_id][:1000*(args.n_slices_max-args.n_slices_min)]
      vec_ids_dict[embed_id] = vec_ids_dict[embed_id][:500500]
      durations_dict[embed_id] = durations_dict[embed_id][:500500]
      landmarks_dict[embed_id] = landmarks_dict[embed_id][:1000]
    
  np.savez(args.exp_dir+"embedding_mats.npz", **embedding_mats)
  np.savez(args.exp_dir+"vec_ids_dict.npz", **vec_ids_dict)
  np.savez(args.exp_dir+"durations_dict.npz", **durations_dict)
  np.savez(args.exp_dir+"landmarks_dict.npz", **landmarks_dict)  
  '''
  print("takes %0.5f s to finish cleaning up dataset" % (time.time() - begin_time))

if start_step <= 2:
  print("Start processing embeddings ...")
  begin_time = time.time()
  embedding_mats = np.load(args.exp_dir+"embedding_mats.npz")
  vec_ids_dict = np.load(args.exp_dir+"vec_ids_dict.npz")
  durations_dict = np.load(args.exp_dir+"durations_dict.npz")
  landmarks_dict = np.load(args.exp_dir+"landmarks_dict.npz")

  embeddings, vec_ids, ids_to_utterance_labels = process_embeddings(embedding_mats, vec_ids_dict)
  np.save(args.exp_dir+"embeddings.npy", embeddings)
  
  with open(args.exp_dir+"vec_ids.json", "w") as f:
    vec_ids_list = [vec_id.tolist() for vec_id in vec_ids]
    json.dump(vec_ids_list, f)

  with open(args.exp_dir+"ids_to_utterance_labels.json", "w") as f:
    json.dump(ids_to_utterance_labels, f)
  print("takes %0.5f s to finish processing embeddings" % (time.time() - begin_time))

if start_step <= 3:
  print("Start training segmentation models")

  begin_time = time.time()
  #embeddings = np.load(args.exp_dir+"embeddings.npy")
  embeddings = np.load("../data/flickr30k/audio_level/embeddings.npy")

  #with open(args.exp_dir+"vec_ids.json", "r") as f:
  with open("../data/flickr30k/audio_level/vec_ids.json", "r") as f:
    vec_ids = json.load(f)  
    vec_ids = [np.asarray(vec_id) for vec_id in vec_ids]

  #with open(args.exp_dir+"ids_to_utterance_labels.json", "r") as f:
  with open("../data/flickr30k/audio_level/ids_to_utterance_labels.json", "r") as f:
    ids_to_utterance_labels = json.load(f)

  durations_dict = np.load("../data/flickr30k/audio_level/durations_dict.npz")
  landmarks_dict = np.load("../data/flickr30k/audio_level/landmarks_dict.npz")

  # Acoustic model parameters
  segmenter = None
  if args.am_class == "fbgmm":
    D = args.embed_dim
    am_class = fbgmm.FBGMM
    am_alpha = 10.
    am_K = 1000
    m_0 = np.zeros(D)
    k_0 = 0.05
    # S_0 = 0.025*np.ones(D)
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
    segmenter = UnigramAcousticWordseg(
      am_class, am_alpha, am_K, am_param_prior, embeddings, vec_ids, 
      ids_to_utterance_labels, 
      durations_dict, landmarks_dict, p_boundary_init=0.5, beta_sent_boundary=-1, n_slices_max=args.n_slices_max
      )
  
  elif args.am_class == "kmeans":
    am_K = args.am_K
    # Initialize model
    '''segmenter = SegmentalKMeansWordseg(am_K, embedding_mats, vec_ids_dict,
      durations_dict, landmarks_dict=landmarks_dict, p_boundary_init=0.5, n_slices_max=args.n_slices_max
      )
    '''
    segmenter = SegmentalKMeansWordseg(am_K, embeddings, vec_ids, ids_to_utterance_labels,
      durations_dict, landmarks_dict=landmarks_dict, p_boundary_init=0.5, n_slices_max=args.n_slices_max
      )
  # Perform sampling
  if args.am_class == "fbgmm":
    record = segmenter.gibbs_sample(5, 3)
    #sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
  else:
    record = segmenter.segment(5, 3)
    sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
  
  print("Take %0.5f s to finish training !" % (time.time() - begin_time))
  np.save("%sboundaries_%s.npy" % (args.exp_dir, args.am_class), segmenter.utterances.boundaries)
  
  if args.am_class == "fbgmm":
    means = []
    for k in range(args.am_K):
      mean = segmenter.acoustic_model.components.rand_k(k)
      means.append(mean)

    np.save(args.exp_dir+"fbgmm_means.npy", np.asarray(means))
  else:
    mean_numerators = segmenter.acoustic_model.mean_numerators
    counts = segmenter.acoustic_model.counts

    np.save(args.exp_dir + "mean_numerators.npy", mean_numerators)
    np.save(args.exp_dir + "counts.npy", counts)

if start_step <= 4:
  boundaries = np.load("%sboundaries_%s.npy" % (args.exp_dir, args.am_class))
  #for i, feat_id in enumerate(sorted(audio_feats.keys(), key=lambda x:int(x.split('_')[-1]))):
  #print boundaries[i]

  # Evaluate (full-coverage boundary scores, concept-level boundary scores, precision for head concept classification)
