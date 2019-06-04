from utils.audio_preprocess import *
from utils.postprocess import *
from utils.clusteval import *
from smt.audio_smt_word_discoverer import *
import argparse
import shutil
import time
import os

# TODO: Add more options
parser = argparse.ArgumentParser()
parser.add_argument('--nmt', help='Use neural encoder-decoder model', action='store_true')
parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory with / at the end')
parser.add_argument('--data_path', type=str, default='data/flickr30k/', help='Data directory with / at the end')
parser.add_argument('--num_mixtures', type=int, default='3', help='Number of mixtures for GMM')
parser.add_argument('--model_dir', type=str, default=None, help='SMT model directory with / at the end')
args = parser.parse_args() 

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
word_level_info_file = '../data/flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json'
train_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_train.npz'
test_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_test.npz'
caption_seqs_file = 'caption_seqs.json' 
phnset = 'pseudophns.npy'
word_align_dir = '/home/lwang114/data/flickr/word_segmentation/'
audio_level_info_file = '../data/flickr30k/audio_level/flickr_bnf_concept_info.json'
word_concept_align_file = '../data/flickr30k/word_level/flickr30k_gold_alignment.json'
gold_align_file = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json' 
  
audio_seq_file = datapath + 'audio_level/flickr_bnf_all_src.npz'
#gold_alignment_nmt_file = datapath + 'audio_level/flickr30k_alignment.ref'
gold_alignment_file = datapath + 'audio_level/flickr30k_gold_alignment.json'
pred_alignment_nmt_file = exp_nmt_dir + 'output/alignment.json'
pred_alignment_smt_prefix = exp_smt_dir + 'flickr30k_pred_alignment'
src_file = datapath + 'audio_level/flickr_bnf_all_src.npz' 
trg_file = datapath + 'audio_level/flickr_bnf_all_trg.txt'

output_path = ''

smt_model_dir = args.model_dir #'smt/models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt'
start = 1
end = 3

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
    model = GMMWordDiscoverer(src_file, trg_file, numMixtures=args.num_mixtures)
    if os.path.isfile(args.model_dir+'model_final_mixture_priors.json'):
      model.trainUsingEM(writeModel=True, mixturePriorFile=args.model_dir+'model_final_mixture_priors.json', transMeanFile=args.model_dir+'model_final_translation_means.json', transVarFile=args.model_dir+'model_final_translation_variances.json')
    else:
      model.trainUsingEM(writeModel=True, modelPrefix=args.model_dir)
    model.printAlignment(out_file_prefix = exp_smt_dir+'flickr30k_pred_alignment')
  
  print('Finish training after %f s !' % (time.time() - start_time))

if start < 3 and end >= 3:
  start_time = time.time()
  print('Start evaluation ...')
  
  pred_aligns = []
  gold_aligns = []

  if args.nmt:
    postproc = XNMTPostprocessor(exp_nmt_dir + 'output/report/')
    postproc.convert_alignment_file(pred_alignment_nmt_file)    
    with open(pred_alignment_nmt_file, 'r') as f:
      pred_aligns = json.load(f)
  else:
    with open(pred_alignment_smt_prefix+'.json' , 'r') as f:   
      pred_aligns = json.load(f)
  
  with open(gold_alignment_file, 'r') as f:
    gold_aligns = json.load(f)

  print('Word IoU: ', word_IoU(pred_aligns, gold_aligns))
  print('Accuracy: ', accuracy(pred_aligns, gold_aligns))
  boundary_retrieval_metrics(pred_aligns, gold_aligns)
  #retrieval_metrics(pred_clsts, gold_clsts)
  print('Finish evaluation after %f s !' % (time.time() - start_time))
