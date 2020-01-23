from hmm_dnn.image_phone_hmm_word_discoverer import *
from hmm_dnn.image_phone_hmm_dnn_word_discoverer import *
from clda.image_phone_word_discoverer import *
from utils.clusteval import *
import argparse
import shutil
import time
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--has_null', help='Include NULL symbol in the image feature', action='store_true')
parser.add_argument('--dataset', choices={'mscoco', 'flickr'}, help='Dataset used for training the model')
parser.add_argument('--feat_type', choices={'synthetic', 'vgg16'}, help='Type of image features')
parser.add_argument('--model_type', choices={'linear', 'gaussian', 'two-layer', 'clda'}, help='Word discovery model type')
parser.add_argument('--momentum', type=float, default=0.3, help='Momentum used for GD iterations (hmm-dnn only)')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate used for GD iterations (hmm-dnn only)')
parser.add_argument('--normalize_vfeat', help='Normalize each image feature to have unit L2 norm', action='store_true')
parser.add_argument('--step_scale', type=float, default=0.1, help='Random jump step scale for simulated annealing (hmm-dnn only)')
parser.add_argument('--width', type=float, default=1., help='width parameter of the radial basis activation function (hmm-dnn only)')
args = parser.parse_args()

if args.dataset == 'mscoco':
  dataDir = 'data/mscoco/'
  speechFeatureFile = dataDir + 'src_mscoco_subset_subword_level_power_law.txt'
  imageConceptFile = dataDir + 'trg_mscoco_subset_subword_level_power_law.txt'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'mscoco_subset_subword_level_concept_gaussian_vectors.npz'
  elif args.feat_type == 'vgg16':
    imageFeatureFile = dataDir + 'mscoco_vgg_penult.npz'
  
  conceptIdxFile = dataDir + 'concept2idx.json'
  goldAlignmentFile = dataDir + 'mscoco_gold_alignment_power_law.json'
  nWords = 65
elif args.dataset == 'flickr':
  dataDir = 'data/flickr30k/phoneme_level/'
  speechFeatureFile = dataDir + 'flickr30k_no_NULL_top_100.txt'
  imageConceptFile = dataDir + 'trg_flickr30k_no_NULL_top_100.txt'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'flickr30k_no_NULL_top_100_gaussian.npz'
  elif args.feat_type == 'vgg16':
    imageFeatureFile = dataDir + 'flickr30k_no_NULL_top_100_vgg_penult.npz'
  
  conceptIdxFile = 'data/flickr30k/concept2idx.json'
  goldAlignmentFile = dataDir + 'flickr30k_no_NULL_top_100_gold_alignment.json'
  nWords = 100

modelConfigs = {
  'has_null': args.has_null, 
  'n_words': nWords, 
  'learning_rate': args.lr,
  'momentum': args.momentum, 
  'normalize_vfeat': args.normalize_vfeat, 
  'step_scale': args.step_scale, 
  'width': args.width
  }

if args.model_type == 'linear' or args.model_type == 'gaussian' or args.model_type == 'two-layer':
  expDir = 'hmm_dnn/exp/mscoco_%s_%s_momentum%.1f_lr%.5f_stepscale%.2f/' % (args.model_type, args.feat_type, modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale']) 
elif args.model_type == 'clda':
  expDir = 'clda/exp/mscoco_%s/' % args.model_type
else:
  raise ValueError('Model type not specified or invalid model type')

modelName = expDir + 'image_phone'
predAlignmentFile = modelName + '_alignment.json'

if not os.path.isdir(expDir):
  print('Create a new directory: ', expDir)
  os.mkdir(expDir)

modelName = expDir + 'image_phone'
print('Experiment directory: ', expDir)
   
tasks = [1, 2, 3]
#-------------------------------#
# Feature extraction for MSCOCO #
#-------------------------------#
if 0 in tasks:    
  vCorpus = {}
  concept2idx = {}
  goldAlignments = {}
  nTypes = 0
  with open(imageConceptFile, 'r') as f:
    vCorpusStr = []
    for line in f:
      vSen = line.strip().split()
      vCorpusStr.append(vSen)
      for vWord in vSen:
        if vWord not in concept2idx:
          concept2idx[vWord] = nTypes
          nTypes += 1
  
  # Generate nTypes different clusters
  imgFeatDim = 2
  permute = True
  centroids = 10 * np.random.normal(size=(nTypes, imgFeatDim)) 
   
  for ex, vSenStr in enumerate(vCorpusStr):
    N = len(vSenStr)
    if permute:
      alignment = np.random.permutation(np.arange(N))
    else:
      alignment = np.arange(N)

    if args.feat_type == 'one-hot':
      vSen = np.zeros((N, nTypes))
      for pos, i_a in enumerate(alignment.tolist()):
        vWord = vSenStr[i_a]
        vSen[pos, concept2idx[vWord]] = 1.
    elif args.feat_type == 'gaussian':
      vSen = np.zeros((N, imgFeatDim))
      for pos, i_a in enumerate(alignment.tolist()):
        vWord = vSenStr[i_a]
        vSen[pos] = centroids[concept2idx[vWord]] + 0.1 * np.random.normal(size=(imgFeatDim,))
      
    vCorpus['arr_'+str(ex)] = vSen
    goldAlignments['arr_'+str(ex)] = {'alignment': alignment.tolist()}

  np.savez(imageFeatureFile, **vCorpus)
  with open(conceptIdxFile, 'w') as f:
    json.dump(concept2idx, f, indent=4, sort_keys=True)
  with open(goldAlignmentFile, 'w') as f:
    json.dump(goldAlignments, f, indent=4, sort_keys=True)

#----------------#
# Model Training #
#----------------#
if 1 in tasks:
  print('Start training the model ...')
  #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
  if args.model_type == 'linear':
    model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
  elif args.model_type == 'gaussian':
    model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
  elif args.model_type == 'two-layer':
    model = ImagePhoneHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)

  model.trainUsingEM(30, writeModel=True, debug=False)
  #model.simulatedAnnealing(numIterations=100, T0=1., debug=False)
  model.printAlignment(modelName+'_alignment', debug=False) 
  print('Finish training the model !')

#------------#
# Evaluation #
#------------#
if 2 in tasks:
  with open(predAlignmentFile, 'r') as f:
    pred_info = json.load(f)
    
  with open(goldAlignmentFile, 'r') as f:
    gold_info = json.load(f)

  with open(conceptIdxFile, 'r') as f:
    concept2idx = json.load(f)
   
  pred, gold = [], []
  for p, g in zip(pred_info, gold_info):
    pred.append(p['image_concepts'])
    if args.dataset == 'flickr':
      gold.append([concept2idx[c] for c in g['image_concepts']]) 
    elif args.dataset == 'mscoco':
      gold.append(g['image_concepts'])
    else:
      raise ValueError('Invalid Dataset')

  cluster_confusion_matrix(gold, pred, file_prefix='image_confusion_matrix')
  cluster_confusion_matrix(gold, pred, alignment=gold_info, file_prefix='audio_confusion_matrix') 
  boundary_retrieval_metrics(pred_info, gold_info)

#---------------#
# Visualization #
#---------------#
if 3 in tasks:
  with open(predAlignmentFile, 'r') as f:
    pred_info = json.load(f)
    
  with open(goldAlignmentFile, 'r') as f:
    gold_info = json.load(f)

  f1_scores = plot_F1_score_histogram(pred_info, gold_info, concept2idx_file=conceptIdxFile, draw_plot=True, out_file=modelName+'_f1_histogram') 
