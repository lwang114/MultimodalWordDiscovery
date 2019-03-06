from utils.preprocess import *
from utils.postprocess import *
from utils.clusteval import *
from smt.smt_word_discoverer import *
import time

# TODO: Add more options

# TODO: Create the path if a path does not exist
datapath = 'data/' 
exp_smt_dir = 'smt/exp/ibm1_phoneme_level_clustering/'
exp_nmt_dir = 'nmt/exp/feb28_clustering_phoneme_level/'
instance_file = datapath + 'flickr30k/bboxes.mat'
image_category_file = datapath + 'imagenet_class_index.json'

raw_alignment_file = datapath + 'flickr30k/flickr30k_phrases.txt'
raw_caption_file = datapath + 'flickr30k/results_20130124.token'
data_info_file = datapath + 'flickr30k/word_level/flickr30k_info_text_concept.json'
phoneme_seq_file = datapath + 'flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json'
phoneme_seq_text_file = datapath + 'flickr30k/phoneme_level/flickr30k.txt'
phoneme_seq_xnmt_text_file = 'flickr30k.txt'
gold_alignment_nmt_file = datapath + 'flickr30k/phoneme_level/flickr30k_alignment.ref'
gold_alignment_smt_file = datapath + 'flickr30k/phoneme_level/flickr30k_gold_alignment.json'
pred_alignment_nmt_file = '' #exp_nmt_dir + 'word'
pred_alignment_smt_prefix = exp_smt_dir + 'flickr30k_pred_alignment'
gold_cluster_file = datapath + 'flickr30k/phoneme_level/flickr30k_gold_clusters.json'
pred_cluster_smt_file = exp_smt_dir + 'flickr30k_pred_cluster.json'
pred_cluster_nmt_file = ''
output_path = ''

smt_model_path = 'smt/models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt'
start = 2
end = 4

if start < 1:
  start_time = time.time() 
  print('Start Preprocessing ...')
  # Preprocessing for SMT
  preproc = Flickr_Preprocessor(instance_file, raw_alignment_file, raw_caption_file, image_path='../../data/flickr30k/flickr30k-images/', category_file=image_category_file)

  #preproc.extract_info(data_info_file)
  preproc.word_to_phoneme(data_info_file, phoneme_seq_file)
  preproc.json_to_text(phoneme_seq_file, phoneme_seq_text_file)
  preproc.json_to_xnmt_text(phoneme_seq_file, phoneme_seq_xnmt_text_file)
  preproc.create_gold_alignment(phoneme_seq_file, gold_alignment_smt_file, is_phoneme=True)
  preproc.create_gold_alignment(phoneme_seq_file, gold_alignment_nmt_file, is_phoneme=True)
  alignment_to_cluster(gold_alignment_smt_file, gold_cluster_file)
  print('Finish Preprocessing after %f s !' % (time.time() - start_time))

if start < 2:
  # Training and testing SMT
  start_time = time.time() 
  print('Start training and generate the alignment ...')
  #if args.model =
  model = IBMModel1(datapath + 'flickr30k/phoneme_level/flickr30k.txt')
  model.initializeWordTranslationProbabilities(smt_model_path)
  #model.trainUsingEM()
  model.printAlignment(pred_alignment_smt_prefix)
  print('Finish training after %f s !' % (time.time() - start_time))

if start < 3:
  start_time = time.time()
  print('Start evaluation ...')
  alignment_to_cluster(pred_alignment_smt_prefix+'.json', pred_cluster_smt_file)

  # Evaluation for SMT
  clsts = []
  classes = []
  clsts_info = []
  classes_info = []
  with open(pred_alignment_smt_prefix+'.json' , 'r') as f:   
    clsts_info = json.load(f)
    for c in clsts_info:
      clsts.append(c['alignment'])

  with open(gold_alignment_smt_file, 'r') as f:
    classes_info = json.load(f) 
    for c in classes_info:
      classes.append(c['alignment'])

  pred_clsts = []
  gold_clsts = []
  with open(pred_cluster_smt_file, 'r') as f:
    pred_clsts = json.load(f)
    
  with open(gold_cluster_file, 'r') as f:
    gold_clsts = json.load(f)
  
  #print('Local clustering purity: ', local_cluster_purity(clsts, classes))
  #print('Global clustering purity: ', cluster_purity(pred_clsts, gold_clsts))
  print('Word IoU: ', word_IoU(clsts_info, classes_info))
  print('Accuracy: ', accuracy(clsts_info, classes_info))
  retrieval_metrics(pred_clsts, gold_clsts)
  print('Finish evaluation after %f s !' % (time.time() - start_time))

# Evaluation for NMT
