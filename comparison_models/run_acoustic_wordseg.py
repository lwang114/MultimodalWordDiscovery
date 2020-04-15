# Driver code to run acoustic word discovery systems
# by Kamper, Livescu and Goldwater 2016
import numpy as np
import json
from segmentalist.segmentalist.unigram_acoustic_wordseg import *
from segmentalist.segmentalist.kmeans_acoustic_wordseg import * 
# from segmentalist.segmentalist.multimodal_kmeans_acoustic_wordseg import * 
# from segmentalist.segmentalist.multimodal_unigram_acoustic_wordseg import *
from bucktsong_segmentalist.downsample.downsample import *   
import segmentalist.segmentalist.fbgmm as fbgmm
# import segmentalist.segmentalist.mfbgmm as mfbgmm 
import segmentalist.segmentalist.gaussian_components_fixedvar as gaussian_components_fixedvar
import segmentalist.segmentalist.kmeans as kmeans
# import segmentalist.segmentalist.mkmeans as mkmeans
from clusteval import *
from postprocess import *
import argparse

random.seed(2)
np.random.seed(2)

logger = logging.getLogger(__name__)
logging.basicConfig(filename="train.log", format="%(asctime)s %(message)s)", level=logging.DEBUG)
i_debug_monitor = 0  # 466  # the index of an utterance which is to be monitored
segment_debug_only = False  # only sample the debug utterance
DEBUG = False
NULL = "NULL"

def downsample(y, n, args):
  if y.shape[1] < n:
    if DEBUG:
      print("y.shape: ", y.shape)
    args.technique = "interpolate"

  y = y[:, :args.mfcc_dim]
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
parser.add_argument("--embed_dim", type=int, default=560, help="Dimension of the embedding vector")
parser.add_argument("--n_slices_min", type=int, default=0, help="Minimum slices between landmarks per segments")
parser.add_argument("--n_slices_max", type=int, default=6, help="Maximum slices between landmarks per segments")
parser.add_argument("--min_duration", type=int, default=0, help="Minimum slices of a segment")
parser.add_argument("--technique", choices={"resample", "interpolate", "rasanen"}, default="resample", help="Downsampling technique")
parser.add_argument("--am_class", choices={"fbgmm", "kmeans", "multimodal-fbgmm", "multimodal-kmeans"}, help="Class of acoustic model")
parser.add_argument("--am_K", type=int, default=65, help="Number of clusters")
parser.add_argument("--exp_dir", type=str, default='./', help="Experimental directory")
parser.add_argument("--feat_type", type=str, choices={"mfcc", "bn", 'kamper'}, help="Acoustic feature type")
parser.add_argument("--mfcc_dim", type=int, default=12, help="Number of the MFCC/delta feature")
parser.add_argument("--landmarks_file", default=None, type=str, help="Npz file with landmark locations")
parser.add_argument('--dataset', choices={'flickr', 'mscoco2k', 'mscoco20k'})
parser.add_argument('--use_null', action='store_true')
args = parser.parse_args()
print(args)

if args.dataset == 'flickr':
  args.use_null = True
  args.landmarks_file = "../data/flickr30k/audio_level/flickr_landmarks_combined.npz"
  if args.feat_type == "bn":
    datapath = "../data/flickr30k/audio_level/flickr_bnf_all_src.npz"
  elif args.feat_type == "mfcc":
    datapath = "../data/flickr30k/audio_level/flickr_mfcc_cmvn_htk.npz"
  else:
    raise ValueError("Please specify the feature type")

  image_concept_file = "../data/flickr30k/audio_level/flickr_bnf_all_trg.txt"
  concept2idx_file = "../data/flickr30k/concept2idx.json"
  pred_boundary_file = "%spred_boundaries.npy" % args.exp_dir
  pred_landmark_segmentation_file = "%sflickr30k_pred_landmark_segmentation.npy" % args.exp_dir
  pred_segmentation_file = "%sflickr30k_pred_segmentation.npy" % args.exp_dir
  gold_segmentation_file = "../data/flickr30k/audio_level/flickr30k_gold_segmentation.json"
  pred_alignment_file = "%sflickr30k_pred_alignment.json" % args.exp_dir
  gold_alignment_file = "../data/flickr30k/audio_level/flickr30k_gold_alignment.json"
elif args.dataset == 'mscoco2k':
  datapath = '../data/mscoco/mscoco2k_kamper_embeddings.npz'
  image_concept_file = '../data/mscoco/mscoco2k_image_captions.txt'
  concept2idx_file = '../data/mscoco/concept2idx.json'
  pred_boundary_file = os.path.join(args.exp_dir, "pred_boundaries.npy")
  pred_segmentation_file = os.path.join(args.exp_dir, "mscoco2k_pred_segmentation.npy")
  pred_landmark_segmentation_file = "%smscoco2k_pred_landmark_segmentation.npy" % args.exp_dir
  gold_segmentation_file = "../data/mscoco/mscoco2k_gold_word_segmentation.npy"
  pred_alignment_file = os.path.join(args.exp_dir, 'mscoco2k_pred_alignment.json')
  gold_alignment_file = '../data/mscoco/mscoco2k_gold_alignment.json'
elif args.dataset == 'mscoco20k':
  datapath = '../data/mscoco/mscoco20k_kamper_embeddings.npz'
  image_concept_file = datapath + 'mscoco20k_image_captions.txt'
  concept2idx_file = '../data/mscoco/concept2idx.json'
  pred_boundary_file = os.path.join(args.exp_dir, "pred_boundaries.npy")
  pred_segmentation_file = os.path.join(args.exp_dir, "mscoco20k_pred_segmentation.npy")
  pred_landmark_segmentation_file = "%smscoco20k_pred_landmark_segmentation.npy" % args.exp_dir
  gold_segmentation_file = "../data/mscoco/mscoco20k_gold_word_segmentation.npy"
  pred_alignment_file = os.path.join(args.exp_dir, 'mscoco20k_pred_alignment.json')
  gold_alignment_file = '../data/mscoco/mscoco20k_gold_alignment.json'
 
# Generate acoustic embeddings, vec_ids_dict and durations_dict 
audio_feats = np.load(datapath)
f = open(image_concept_file, "r")
image_concepts = []
for line in f:
  image_concepts.append(line.strip().split())
f.close()

with open(concept2idx_file, "r") as f:
  concept2idx = json.load(f)

embedding_mats = {}
concept_ids = []
vec_ids_dict = {}
durations_dict = {}
landmarks_dict = {}
if args.landmarks_file: 
  landmarks_dict = np.load(args.landmarks_file)
  landmark_ids = sorted(landmarks_dict, key=lambda x:int(x.split('_')[-1]))
else:
  landmark_ids = []

start_step = 4
if start_step == 0:
  print("Start extracting acoustic embeddings")
  begin_time = time.time()
  
  for i_ex, feat_id in enumerate(sorted(audio_feats.keys(), key=lambda x:int(x.split('_')[-1]))):
    # XXX
    if i_ex > 5:
      break
    print (feat_id)
    feat_mat = audio_feats[feat_id]
    if feat_mat.shape[0] > 1000:
      feat_mat = feat_mat[:1000, :args.mfcc_dim]
    else:
      feat_mat = feat_mat[:, :args.mfcc_dim]

    if not args.landmarks_file:
      n_slices = feat_mat.shape[0]
      landmarks_dict[feat_id] = np.arange(n_slices)
      landmark_ids.append(feat_id)
    else:   
      n_slices = len(landmarks_dict[landmark_ids[i_ex]]) 
    
    feat_dim = args.mfcc_dim 
    assert args.embed_dim % feat_dim == 0   
    embed_mat = np.zeros(((args.n_slices_max - max(args.n_slices_min, 1) + 1)*n_slices, args.embed_dim))
    if args.am_class.split("-")[0] == "multimodal":
      concept_ids_i = [[] for _ in range((args.n_slices_max - max(args.n_slices_min, 1) + 1)*n_slices)] 
    vec_ids = -1 * np.ones((n_slices * (1 + n_slices) / 2,))
    durations = np.nan * np.ones((n_slices * (1 + n_slices) / 2,))

    i_embed = 0        
    
    # Store the vec_ids using the mapping i_embed = end * (end - 1) / 2 + start (following unigram_acoustic_wordseg.py)
    for cur_start in range(n_slices):
        for cur_end in range(cur_start + max(args.n_slices_min, 1), min(n_slices, cur_start + args.n_slices_max)):
            cur_end += 1
            t = cur_end
            i = t*(t - 1)/2
            vec_ids[i + cur_start] = i_embed
            n_down_slices = args.embed_dim / feat_dim
            start_frame, end_frame = landmarks_dict[landmark_ids[i_ex]][cur_start], landmarks_dict[landmark_ids[i_ex]][cur_end-1]
            if end_frame - start_frame == 1:
              embed_mat[i_embed] = np.repeat(feat_mat[start_frame:end_frame], n_down_slices)
            else:
              embed_mat[i_embed] = downsample(feat_mat[start_frame:end_frame].T, n_down_slices, args) 
            #embed_mat.append(downsample(feat_mat[cur_start:cur_end].T, n_down_slices, args))
            if args.am_class.split("-")[0] == "multimodal":
              concept_ids_i[i_embed] = [concept2idx[NULL]] + [concept2idx[c] for c in image_concepts[i_ex]]
              #concept_ids_i.append([concept2idx[NULL]] + [concept2idx[c] for c in image_concepts[i_ex]])
           
            durations[i + cur_start] = end_frame - start_frame
            i_embed += 1 

    vec_ids_dict[feat_id] = vec_ids
    embedding_mats[feat_id] = embed_mat
    durations_dict[feat_id] = durations 

  np.savez(args.exp_dir+"embedding_mats.npz", **embedding_mats)
  np.savez(args.exp_dir+"vec_ids_dict.npz", **vec_ids_dict)
  np.savez(args.exp_dir+"durations_dict.npz", **durations_dict)
  np.savez(args.exp_dir+"landmarks_dict.npz", **landmarks_dict)  
      
  print("Take %0.5f s to finish extracting embedding vectors !" % (time.time()-begin_time))
  if args.am_class.split("-")[0] == "multimodal":
    with open(args.exp_dir+"image_concepts.json", "w") as f:
      json.dump(concept_ids, f, indent=4, sort_keys=True)
    
    with open(args.exp_dir+"concept_names.json", "w") as f:
      concept_names = [c for c, i in sorted(concept2idx.items(), key=lambda x:x[1])]
      json.dump(concept_names, f, indent=4, sort_keys=True)

if start_step <= 1:
  begin_time = time.time()
  if args.am_class.split("-")[0] == "multimodal":
    with open(args.exp_dir+"image_concepts.json", "r") as f:
      concepts = json.load(f) 
    with open(args.exp_dir+"concept_names.json", "r") as f:
      concept_names = json.load(f)
        
  embedding_mats = np.load(args.exp_dir+'embedding_mats.npz')
  vec_ids_dict = np.load(args.exp_dir+'vec_ids_dict.npz')
  durations_dict = np.load(args.exp_dir+"durations_dict.npz")
  landmarks_dict = np.load(args.exp_dir+"landmarks_dict.npz")
  # Ensure the landmark ids and utterance ids are the same
  if args.feat_type == "bn":
    landmarks_ids = sorted(landmarks_dict, key=lambda x:int(x.split('_')[-1]))
    new_landmarks_dict = {}
    for lid, uid in zip(landmarks_ids, ids_to_utterance_labels):
      new_landmarks_dict[uid] = landmarks_dict[lid]
    np.savez(args.exp_dir+"new_landmarks_dict.npz", **new_landmarks_dict)
    landmarks_dict = np.load(args.exp_dir+"new_landmarks_dict.npz") 

  print("Start training segmentation models")
  # Acoustic model parameters
  segmenter = None
  if args.am_class == "fbgmm":
    D = args.embed_dim
    am_class = fbgmm.FBGMM
    am_alpha = 10.
    am_K = args.am_K
    m_0 = np.zeros(D)
    k_0 = 0.05
    # S_0 = 0.025*np.ones(D)
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
    segmenter = UnigramAcousticWordseg(
      am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict, 
      durations_dict, landmarks_dict, p_boundary_init=0.1, beta_sent_boundary=-1, 
      n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max
      ) 
  # TODO Fix inputs
  elif args.am_class == "multimodal-fbgmm":
    D = args.embed_dim
    am_class = mfbgmm.MultimodalFBGMM
    am_alpha = 10.
    m_0 = np.zeros(D)
    k_0 = 0.05
    # S_0 = 0.025*np.ones(D)
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
    segmenter = MultimodalUnigramAcousticWordseg(
      am_class, am_alpha, am_param_prior, concepts, concept_names, embeddings, vec_ids, 
      ids_to_utterance_labels, 
      durations_dict, landmarks_dict, p_boundary_init=0.1, beta_sent_boundary=-1, 
      n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max
      ) 
  # TODO Fix inputs
  elif args.am_class == "kmeans":
    am_K = args.am_K 
    segmenter = SegmentalKMeansWordseg(am_K, embeddings, vec_ids, ids_to_utterance_labels,
      durations_dict, landmarks_dict=landmarks_dict, p_boundary_init=0.1, n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max
      )
  elif args.am_class == "multimodal-kmeans":
    segmenter = MultimodalSegmentalKMeansWordseg(concepts, concept_names, embeddings, vec_ids, ids_to_utterance_labels,
      durations_dict, landmarks_dict=landmarks_dict, p_boundary_init=0.1, n_slices_min=args.n_slices_min, n_slices_max=args.n_slices_max
      )
  else:
    raise ValueError("am_class %s is not supported" % args.am_class)

  # Perform sampling
  if args.am_class.split("-")[-1] == "fbgmm":
    record = segmenter.gibbs_sample(30, 3, anneal_schedule="linear", anneal_gibbs_am=True)
    #sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
  else:
    record = segmenter.segment(30, 3)
    sum_neg_len_sqrd_norm = record["sum_neg_len_sqrd_norm"] 
  
  print("Take %0.5f s to finish training !" % (time.time() - begin_time))
  np.save("%spred_boundaries.npy" % args.exp_dir, segmenter.utterances.boundaries)
  
  if args.am_class.split("-")[-1] == "fbgmm":
    means = []
    for k in range(segmenter.acoustic_model.components.K_max):
      mean = segmenter.acoustic_model.components.rand_k(k)
      means.append(mean)
    np.save(args.exp_dir+"fbgmm_means.npy", np.asarray(means))
  else:
    mean_numerators = segmenter.acoustic_model.components.mean_numerators
    counts = segmenter.acoustic_model.components.counts
    np.save(args.exp_dir + "mean_numerators.npy", mean_numerators)
    np.save(args.exp_dir + "counts.npy", counts)

  if args.am_class.split("-")[0] == "multimodal":
    segmenter.get_alignments(out_file_prefix=args.exp_dir+"flickr30k_pred_alignment")

if start_step <= 4:
  convert_boundary_to_segmentation(pred_boundary_file, pred_landmark_segmentation_file)
  if args.landmarks_file:
    convert_landmark_segment_to_10ms_segmentation(pred_landmark_segmentation_file, args.landmarks_file, pred_segmentation_file)
  else:
    convert_landmark_segment_to_10ms_segmentation(pred_landmark_segmentation_file, os.path.join(args.exp_dir, "landmarks_dict.npz"), pred_segmentation_file)  
  pred_segs = np.load(pred_segmentation_file, encoding="latin1", allow_pickle=True)
  gold_segs = np.load(gold_segmentation_file, encoding="latin1", allow_pickle=True)
  segmentation_retrieval_metrics(pred_segs, gold_segs)    
  
  # TODO: Make the word IoU work later
  if args.am_class.split('-')[0] == 'multimodal':
    with open(pred_alignment_file, "w") as f:
      pred_aligns = json.load(f)
    with open(gold_alignment_file, "w") as f:
      gold_aligns = json.load(f)
    print('Accuracy: ', accuracy(pred_aligns, gold_aligns))
    boundary_retrieval_metrics(pred_aligns, gold_aligns)
    #retrieval_metrics(pred_clsts, gold_clsts)
    print('Word IoU: ', word_IoU(pred_aligns, gold_aligns))
    print('Finish evaluation after %f s !' % (time.time() - start_time))
