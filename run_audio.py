from utils.audio_preprocess import *
from utils.postprocess import *
from utils.clusteval import *
from utils.plot import *
from smt.audio_gmm_word_discoverer import *
from smt.audio_kmeans_word_discoverer import *
from smt.audio_segembed_kmeans_word_discoverer import *
import argparse
import shutil
import time
import os
import logging

# TODO: Add more options
parser = argparse.ArgumentParser()
parser.add_argument('--nmt', help='Use neural encoder-decoder model', action='store_true')
parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory with / at the end')
parser.add_argument('--data_path', type=str, default='data/flickr30k/', help='Data directory with / at the end')
parser.add_argument('--num_mixtures', type=int, default='3', help='Number of mixtures for GMM')
parser.add_argument('--model_dir', type=str, default='', help='SMT model directory with / at the end')
parser.add_argument('--smt_model', choices=['gmm', 'kmeans', 'segembed-kmeans'], default='gmm', help='Type of SMT model')
parser.add_argument('--embed_dim', type=int, default=None, help='Acoustic embedding dimension; used only in segmental embedded models')
parser.add_argument('--min_word_len', type=int, default=None, help='Minimal number of feature frames per word; used only in segmental embedded models')
parser.add_argument('--max_word_len', type=int, default=None, help='Maximal number of feature frames per word; used only in segmental embedded models')
parser.add_argument('--feat_type', choices={"mfcc", "bn"})
parser.add_argument("--context_width", type=int, default=0, help="Width of the context window in acoustic feature; used only in frame-level models")
parser.add_argument("--boundary_file", type=str, default=None, help="Pre-segmentation for the segmental models")
args = parser.parse_args() 

logging.basicConfig(filename="run_audio.log", format="%(asctime)s %(message)s", level=DEBUG)

# TODO: Create the path if a path does not exist
datapath = args.data_path 
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
word_level_info_file = 'data/flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json'
train_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_train.npz'
test_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_test.npz'
caption_seqs_file = 'caption_seqs.json' 
phnset = 'pseudophns.npy'
word_align_dir = '/home/lwang114/data/flickr/word_segmentation/'
audio_level_info_file = 'data/flickr30k/audio_level/flickr_bnf_concept_info.json'
word_concept_align_file = 'data/flickr30k/word_level/flickr30k_gold_alignment.json'
gold_align_file = 'data/flickr30k/audio_level/flickr30k_gold_alignment.json'
gold_segmentation_file = "../data/flickr30k/audio_level/flickr30k_gold_segmentation_mfcc.json"
src_feat2wavs_file = "data/flickr30k/audio_level/flickr_mfcc_feat2wav.json" 
trg_feat2wavs_file = "data/flickr30k/audio_level/flickr30k_gold_alignment.json_feat2wav.json"
  
audio_seq_file = datapath + 'audio_level/flickr_bnf_all_src.npz'
#gold_alignment_nmt_file = datapath + 'audio_level/flickr30k_alignment.ref'
gold_alignment_file = datapath + 'audio_level/flickr30k_gold_alignment.json'
pred_alignment_nmt_file = exp_nmt_dir + 'output/alignment.json'
pred_alignment_smt_prefix = exp_smt_dir + 'flickr30k_pred_alignment'
pred_segmentation_file = exp_smt_dir + "flickr30k_pred_segmentation.npy"
if args.feat_type == "mfcc":
  src_file = datapath + 'audio_level/flickr_mfcc_cmvn.npz'
elif args.feat_type == "bn":
  src_file = datapath + 'audio_level/flickr_bnf_all_src.npz' 
else:
  raise ValueError("feature type does not exist")
trg_file = datapath + 'audio_level/flickr_bnf_all_trg.txt'

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
      model = SegEmbedKMeansWordDiscoverer(args.num_mixtures, src_file, trg_file, 
                                          embedDim=args.embed_dim,
                                          minWordLen=args.min_word_len, maxWordLen=args.max_word_len,
                                          boundaryFile=args.boundary_file)
       
    if os.path.isfile(args.model_dir+'model_final*'):
      if args.smt_model == "gmm":
        model.trainUsingEM(writeModel=True, mixturePriorFile=args.model_dir+'model_final_mixture_priors.json', 
                          transMeanFile=args.model_dir+'model_final_translation_means.json', 
                          transVarFile=args.model_dir+'model_final_translation_variances.json')
      elif args.smt_model == 'kmeans' or self.smt_model == 'segembed_kmeans':
        model.trainUsingEM(writeModel=True, centroidFile=args.model_dir+'model_final.json')
    else:
      args.model_dir = args.exp_dir
      model.trainUsingEM(writeModel=True, modelPrefix=args.model_dir)
    model.printAlignment(exp_smt_dir+'flickr30k_pred_alignment')
  
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
    
    with open(gold_alignment_file, "r") as f:
      gold_aligns = json.load(f)
  
    if args.feat_type == "mfcc":
      resample_alignment(pred_alignment_smt_prefix+".json", src_feat2wavs_file, trg_feat2wavs_file, pred_alignment_smt_prefix+"_resample.json")
      with open(pred_alignment_smt_prefix+'_resample.json' , 'r') as f:   
        pred_aligns = json.load(f)
  
  n_ex = len(pred_aligns) 

  if args.smt_model.split("-")[0] == "segembed":
    pred_segs = np.load(pred_segmentation_file)
    gold_segs = np.load(gold_segmentation_file)
    segmentation_retrieval_metrics(pred_segs, gold_segs)    
  
  # TODO: Make the word IoU work later
  print('Accuracy: ', accuracy(pred_aligns, gold_aligns))
  boundary_retrieval_metrics(pred_aligns, gold_aligns)
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
  rand_ids = np.random.randint(0, n_ex-1, 10).tolist()
  if args.nmt:
    generate_nmt_attention_plots(pred_alignment_file, indices=rand_ids, out_dir=args.exp_dir + "attention_plot_")
  else:
    generate_smt_alignprob_plots(pred_alignment_file, indices=rand_ids, out_dir=args.exp_dir + "align_prob_plot_")
  print("Finishing drawing attention plots after %f s !" % (time.time() - start_time))
  
 
