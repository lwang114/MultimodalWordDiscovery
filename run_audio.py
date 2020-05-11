from utils.audio_preprocess import *
from utils.postprocess import *
from utils.clusteval import *
from utils.plot import *
from smt.audio_gmm_word_discoverer import *
from smt.audio_kmeans_word_discoverer import *
from smt.audio_segembed_kmeans_word_discoverer import *
from smt.audio_segembed_gmm_word_discoverer import *
from hmm.audio_segembed_hmm_word_discoverer import *
from hmm.audio_hmm_word_discoverer import *
import argparse
import shutil
import time
import os
import logging

# TODO: Make the number of iterations and model_dir option working
parser = argparse.ArgumentParser()
parser.add_argument('--nmt', help='Use neural encoder-decoder model', action='store_true')
parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory with / at the end')
# parser.add_argument('--data_path', type=str, default='data/flickr30k/', help='Data directory with / at the end')
parser.add_argument('--dataset', choices=['mscoco2k', 'mscoco20k', 'flickr'])
parser.add_argument('--num_mixtures', type=int, default=1, help='Number of mixtures for GMM')
parser.add_argument('--model_dir', type=str, default='', help='SMT model directory with / at the end')
parser.add_argument('--smt_model', choices=['gmm', 'kmeans', 'segembed-kmeans', 'segembed-gmm', 'segembed-hmm'], default='gmm', help='Type of SMT model')
parser.add_argument('--embed_dim', type=int, default=120, help='Acoustic embedding dimension; used only in segmental embedded models')
parser.add_argument('--min_word_len', type=int, default=None, help='Minimal number of feature frames per word; used only in segmental embedded models')
parser.add_argument('--max_word_len', type=int, default=None, help='Maximal number of feature frames per word; used only in segmental embedded models')
parser.add_argument('--feat_type', choices={"mfcc", "bn", 'kamper'})
parser.add_argument('--frame_dim', type=int, default=12, help="Dimension of the feature frame")
parser.add_argument("--context_width", type=int, default=0, help="Width of the context window in acoustic feature; used only in frame-level models")
parser.add_argument("--preseg_file", type=str, default=None, help="Pre-segmentation for the segmental models")
parser.add_argument("--temperature", type=float, default=10, help="temperature for the attention/align probability matrix plot")
parser.add_argument("--num_iterations", type=int, default=20, help="Number of iterations for training")
parser.add_argument("--max_feat_len", type=int, default=2000, help="Maximal number of feature frames for an utterance")
parser.add_argument('--use_null', help='Use NULL concept', action='store_true')
args = parser.parse_args() 


logging.basicConfig(filename="run_audio.log", format="%(asctime)s %(message)s", level=DEBUG)

if args.dataset == 'flickr':
  datapath = 'data/flickr30k/'
elif args.dataset == 'mscoco20k':
  datapath = 'data/mscoco/'
elif args.dataset == 'mscoco2k':
  datapath = 'data/mscoco/'
else:
  raise NotImplementedError

exp_smt_dir = 'smt/exp/gmm_audio_level_clustering/'
exp_nmt_dir = 'nmt/exp/nmt_audio_level_clustering/'

if args.exp_dir:
  if not os.path.isdir(args.exp_dir):
    os.mkdir(args.exp_dir)
  if args.nmt:
    exp_nmt_dir = args.exp_dir
  else:
    exp_smt_dir = args.exp_dir

# Dataset specific variables (TODO: make it more general)
if args.dataset == 'flickr':
  args.use_null = True 
  word_level_info_file = 'data/flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json'
  train_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_train.npz'
  test_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_test.npz'
  caption_seqs_file = 'caption_seqs.json' 
  phnset = 'pseudophns.npy'
  word_align_dir = '/home/lwang114/data/flickr/word_segmentation/'
  audio_level_info_file = 'data/flickr30k/audio_level/flickr_bnf_concept_info.json'
  word_concept_align_file = 'data/flickr30k/word_level/flickr30k_gold_alignment.json'
  gold_align_file = 'data/flickr30k/audio_level/flickr30k_gold_alignment.json'
  gold_segmentation_file = "data/flickr30k/audio_level/flickr30k_gold_segmentation_mfcc_htk.npy"
  #src_feat2wavs_file = "data/flickr30k/audio_level/flickr_mfcc_cmvn_htk_feat2wav.json" 
  src_feat2wavs_file = "data/flickr30k/audio_level/flickr_mfcc_cmvn_htk_feat2wav.json" 
  trg_feat2wavs_file = "data/flickr30k/audio_level/flickr30k_gold_alignment.json_feat2wav.json"

  audio_seq_file = datapath + 'audio_level/flickr_bnf_all_src.npz'
  #gold_alignment_nmt_file = datapath + 'audio_level/flickr30k_alignment.ref'
  gold_alignment_file = datapath + 'audio_level/flickr30k_gold_alignment.json'
  pred_alignment_nmt_file = exp_nmt_dir + 'output/alignment.json'
  pred_alignment_smt_prefix = exp_smt_dir + 'flickr30k_pred_alignment'
  pred_boundary_file = exp_smt_dir + "pred_boundaries.npy"
  landmarks_file = datapath + "audio_level/flickr_landmarks.npz" 
  pred_landmark_segmentation_file = exp_smt_dir + "flickr30k_pred_landmark_segmentation.npy"
  pred_segmentation_file = exp_smt_dir + "flickr30k_pred_segmentation.npy"

  if args.feat_type == "mfcc":
    src_file = datapath + 'audio_level/flickr_mfcc_cmvn_htk.npz'
    #src_file = datapath + 'audio_level/flickr_mfcc_cmvn.npz'
  elif args.feat_type == "bn":
    src_file = datapath + 'audio_level/flickr_bnf_all_src.npz' 
  else:
    raise ValueError("feature type does not exist")
  trg_file = datapath + 'audio_level/flickr_bnf_all_trg.txt'
elif args.dataset == 'mscoco20k':
  # XXX
  args.embed_dim = 560
  
  landmarks_file = datapath + "mscoco20k_landmarks.npz" 
  pred_landmark_segmentation_file = exp_smt_dir + "mscoco20k_pred_landmark_segmentation.npy"
  pred_segmentation_file = exp_smt_dir + "mscoco20k_pred_segmentation.npy"
  gold_alignment_file = datapath + 'mscoco20k_gold_alignment.json'
  pred_alignment_smt_prefix = exp_smt_dir + 'mscoco20k_pred_alignment' 
  pred_boundary_file = exp_smt_dir + "pred_boundaries.npy"

  if args.feat_type == 'kamper':
    src_file = datapath + 'mscoco20k_kamper_embeddings.npz'
    trg_file = datapath + 'mscoco20k_image_captions.txt'
  elif args.feat_type == 'mfcc':
    # TODO Generate this
    src_file = datapath + 'mscoco20k_mfcc_unsegmented.npz'
    trg_file = datapath + 'mscoco20k_image_captions.txt'
    raise ValueError("feature type does not exist")
elif args.dataset == 'mscoco2k':
  # XXX
  args.embed_dim = 560
  landmarks_file = datapath + "mscoco2k_landmarks.npz" 
  pred_landmark_segmentation_file = exp_smt_dir + "mscoco2k_pred_landmark_segmentation.npy"
  pred_segmentation_file = exp_smt_dir + "mscoco2k_pred_segmentation.npy"
  gold_alignment_file = datapath + 'mscoco2k_gold_alignment.json'
  pred_alignment_smt_prefix = exp_smt_dir + 'mscoco2k_pred_alignment' 
  pred_boundary_file = exp_smt_dir + "pred_boundaries.npy"

  if args.feat_type == 'kamper':
    src_file = datapath + 'mscoco2k_kamper_embeddings.npz'
    trg_file = datapath + 'mscoco2k_image_captions.txt'
  elif args.feat_type == 'mfcc':
    src_file = datapath + 'mscoco2k_mfcc_unsegmented.npz'
    trg_file = datapath + 'mscoco2k_image_captions.txt'
  else:
    raise ValueError("feature type does not exist")

output_path = ''
smt_model_dir = args.model_dir #'smt/models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt'
start = 1
end = 4

if start < 1 and end >= 1:
  start_time = time.time() 
  print('Start Preprocessing ...')
  # Preprocessing for SMT
  preproc = FlickrBottleneckPreprocessor(train_file, test_file, word_level_info_file, word_align_dir, phone_centroids_file=phnset, caption_seqs_file=caption_seqs_file)
  #bn_preproc.extract_info(out_file=out_file)
  #bn_preproc.label_captions()
  preproc.create_gold_alignment(audio_level_info_file, word_concept_align_file, out_file=gold_align_file)
  preproc.json_to_xnmt_format(datapath + 'audio_level/' + audio_level_info_file, gold_align_file, train_test_split=False)
 
  print('Finish Preprocessing after %f s !' % (time.time() - start_time))

if start < 2 and end >= 2:
  # Training and testing SMT
  start_time = time.time() 
  print('Start training and generate the alignment ...')
  
  # TODO Cross validation
  if args.nmt:
    print('Please train with XNMT and make sure the output files have been generated in the output/ dir')
  else:
    if args.smt_model == "gmm":
      model = GMMWordDiscoverer(args.num_mixtures, src_file, trg_file,
                                contextWidth=args.context_width)
    elif args.smt_model == "kmeans":
      model = KMeansWordDiscoverer(args.num_mixtures, src_file, trg_file, 
                                  contextWidth=args.context_width)
    elif args.smt_model == "segembed-kmeans":
      # TODO
      acoustic_model = KMeansWordDiscoverer
      model = SegEmbedKMeansWordDiscoverer(acoustic_model, args.num_mixtures, src_file, trg_file, 
                                          embedDim=args.embed_dim,
                                          minWordLen=args.min_word_len, maxWordLen=args.max_word_len,
                                          boundaryFile=args.preseg_file)
    elif args.smt_model == "segembed-gmm":
      model_dir = None
      if os.path.isfile(args.model_dir+"model_final*"): 
        model_dir = args.model_dir
      acoustic_model = GMMWordDiscoverer
      model = SegEmbedGMMWordDiscoverer(acoustic_model, args.num_mixtures, args.frame_dim, src_file, trg_file,
                                      embedDim=args.embed_dim,
                                      modelDir = model_dir,
                                      minWordLen=args.min_word_len, maxWordLen=args.max_word_len,
                                      useNULL=args.use_null,
                                      landmarkFile=args.preseg_file)
    elif args.smt_model == "segembed-hmm":
      model_dir = None
      if os.path.isfile(args.model_dir+"model_final*"): 
        model_dir = args.model_dir
      acoustic_model = AudioHMMWordDiscoverer
      model = SegEmbedHMMWordDiscoverer(acoustic_model, args.num_mixtures, args.frame_dim, args.embed_dim, 
                                      src_file, trg_file,
                                      modelDir = model_dir,
                                      minWordLen=args.min_word_len, maxWordLen=args.max_word_len,
                                      landmarkFile=args.preseg_file)
      
    if os.path.isfile(args.model_dir+'model_final*'):
      if args.smt_model == "gmm":
        model.trainUsingEM(numIterations=args.num_iterations, writeModel=True, mixturePriorFile=args.model_dir+'model_final_mixture_priors.json', 
                          transMeanFile=args.model_dir+'model_final_translation_means.json', 
                          transVarFile=args.model_dir+'model_final_translation_variances.json')
      elif args.smt_model == 'kmeans' or self.smt_model == 'segembed_kmeans':
        model.trainUsingEM(numIterations=args.num_iterations, writeModel=True, centroidFile=args.model_dir+'model_final.json')
    else:
      args.model_dir = args.exp_dir
      model.trainUsingEM(numIterations=args.num_iterations, writeModel=False, modelPrefix=args.model_dir)
    
    model.printAlignment(pred_alignment_smt_prefix)
  print('Finish training after %f s !' % (time.time() - start_time))

if start < 3 and end >= 3:
  start_time = time.time()
  print("Start evaluation ...")
  
  pred_aligns = []
  gold_aligns = []

  if args.nmt:
    postproc = XNMTPostprocessor(exp_nmt_dir + 'output/report/')
    postproc.convert_alignment_file(pred_alignment_nmt_file)    
    with open(pred_alignment_nmt_file, 'r') as f:
      pred_aligns = json.load(f)
    
    if args.feat_type == "mfcc":
      resample_alignment(pred_alignment_nmt_file, src_feat2wavs_file, trg_feat2wavs_file, pred_alignment_nmt_file+"_resample.json")
      with open(pred_alignment_nmt_file+"_resample.json" , "r") as f:   
        pred_aligns = json.load(f)
  else:
    with open(pred_alignment_smt_prefix+".json" , "r") as f:   
      pred_aligns = json.load(f)
  
    if args.dataset == 'flickr' and args.feat_type == "mfcc":
      resample_alignment(pred_alignment_smt_prefix+".json", src_feat2wavs_file, trg_feat2wavs_file, pred_alignment_smt_prefix+"_resample.json")
      with open(pred_alignment_smt_prefix+'_resample.json' , 'r') as f:   
        pred_aligns = json.load(f)
  n_ex = len(pred_aligns)

  with open(gold_alignment_file, "r") as f:
    gold_aligns = json.load(f)  

  '''if args.smt_model.split("-")[0] == "segembed":
    convert_boundary_to_segmentation(pred_boundary_file, pred_landmark_segmentation_file)
    convert_landmark_to_10ms_segmentation(pred_landmark_segmentation_file, landmarks_file, pred_segmentation_file)
    pred_segs = np.load(pred_segmentation_file, encoding="latin1")
    gold_segs = np.load(gold_segmentation_file, encoding="latin1")
    segmentation_retrieval_metrics(pred_segs, gold_segs)    
  '''
  # TODO: Make the word IoU work later
  print('Accuracy: ', accuracy(pred_aligns, gold_aligns, max_len=args.max_feat_len))
  boundary_retrieval_metrics(pred_aligns, gold_aligns, max_len=args.max_feat_len)
  #retrieval_metrics(pred_clsts, gold_clsts)
  print('Word IoU: ', word_IoU(pred_aligns, gold_aligns))
  print('Finish evaluation after %f s !' % (time.time() - start_time))
  
if start < 4 and end >= 4:
  start_time = time.time()
  print("Generating plots ...")
  if not args.nmt and args.feat_type == "mfcc":
    pred_alignment_file = pred_alignment_smt_prefix+"_resample.json"
  elif not args.nmt:
    pred_alignment_file = pred_alignment_smt_prefix+".json"
  elif args.feat_type == "mfcc":
    pred_alignment_file = pred_alignment_nmt_file+"_resample.json"
  else:
    pred_alignment_file = pred_alignment_nmt_file
  
  top_classes, top_freqs = plot_word_len_distribution(pred_alignment_file, args.exp_dir+"length_distribution", draw_plot=False, phone_level=False)
  #plt.plot(top_classes[:50], top_freqs[:50])
  #print(np.sum(top_freqs))
  print("Finishing drawing length distribution plots after %f s !" % (time.time() - start_time))
  
  start_time = time.time() 
  plot_avg_roc(pred_alignment_file, gold_alignment_file, concept2idx="data/flickr30k/concept2idx.json", out_file=args.exp_dir + "roc")
  print("Finishing drawing roc plots after %f s !" % (time.time() - start_time))
  
  start_time = time.time()
  n_ex = 6000
  #rand_ids = np.random.randint(0, n_ex-1, 10).tolist()
  rand_ids = np.arange(1).tolist()
  if args.nmt:
    generate_nmt_attention_plots(pred_alignment_file, indices=rand_ids, out_dir=args.exp_dir + "attention_plot_")
  else:
    generate_smt_alignprob_plots(pred_alignment_file, indices=rand_ids, out_dir=args.exp_dir + "align_prob_plot_", log_prob=True, T=args.temperature)
  print("Finishing drawing attention plots after %f s !" % (time.time() - start_time))  
