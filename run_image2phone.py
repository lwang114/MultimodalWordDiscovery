from hmm_dnn.image_phone_hmm_word_discoverer import *
from hmm_dnn.image_phone_hmm_dnn_word_discoverer import *
from hmm_dnn.image_phone_gaussian_hmm_word_discoverer import *
from clda.image_phone_word_discoverer import *
from utils.clusteval import *
from utils.plot import *
import argparse
import shutil
import time
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--has_null', help='Include NULL symbol in the image feature', action='store_true')
parser.add_argument('--dataset', choices={'mscoco2k', 'mscoco20k', 'flickr'}, help='Dataset used for training the model')
parser.add_argument('--feat_type', choices={'synthetic', 'vgg16_penult', 'res34'}, help='Type of image features')
parser.add_argument('--model_type', choices={'linear', 'gaussian', 'two-layer', 'clda'}, default='gaussian', help='Word discovery model type')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum used for GD iterations (hmm-dnn only)')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate used for GD iterations (hmm-dnn only)')
parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension (two-layer hmm-dnn only)')
parser.add_argument('--normalize_vfeat', help='Normalize each image feature to have unit L2 norm', action='store_true')
parser.add_argument('--step_scale', type=float, default=0.1, help='Random jump step scale for simulated annealing (hmm-dnn only)')
parser.add_argument('--width', type=float, default=1., help='width parameter of the radial basis activation function (hmm-dnn only)')
parser.add_argument('--image_posterior_weights_file', type=str, default=None, help='Pretrained weights for the image posteriors')
parser.add_argument('--date', type=str, default='', help='Date of starting the experiment')
args = parser.parse_args()

if args.dataset == 'mscoco2k':
  dataDir = 'data/mscoco/'
  speechFeatureFile = 'tdnn/exp/blstm2_mscoco_train_sgd_lr_0.00010_feb28/phone_features_discrete.txt' # XXX dataDir + 'src_mscoco_subset_subword_level_power_law.txt'
  imageConceptFile = dataDir + 'trg_mscoco_subset_subword_level_power_law.txt'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'mscoco_subset_subword_level_concept_gaussian_vectors.npz'
  elif args.feat_type == 'vgg16_penult':
    imageFeatureFile = dataDir + 'mscoco_vgg_penult.npz'
  elif args.feat_type == 'res34':
    imageFeatureFile = dataDir + 'mscoco_subset_2k_res34_embed512dim.npz'

  conceptIdxFile = dataDir + 'concept2idx.json'
  goldAlignmentFile = dataDir + 'mscoco_gold_alignment_power_law.json'
  nWords = 65
elif args.dataset == 'mscoco20k':
  dataDir = 'data/mscoco/'
  speechFeatureFile = dataDir + 'src_mscoco_subset_130k_power_law_phone_captions.txt'
  imageConceptFile = dataDir + 'trg_mscoco_subset_130k_power_law_phone_captions.txt'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'mscoco_subset_130k_power_law_concept_gaussian_vectors.npz'
  elif args.feat_type == 'vgg16_penult':
    imageFeatureFile = dataDir + 'mscoco_subset_130k_vgg16_penult.npz'
  elif args.feat_type == 'res34':
    imageFeatureFile = dataDir + 'mscoco_subset_130k_res34_embed512dim.npz'

  conceptIdxFile = dataDir + 'concept2idx.json'
  # TODO: Generate this
  goldAlignmentFile = dataDir + 'mscoco_gold_alignment_130k_power_law.json'
  nWords = 65
elif args.dataset == 'flickr':
  dataDir = 'data/flickr30k/phoneme_level/'
  speechFeatureFile = dataDir + 'flickr30k_no_NULL_top_100.txt'
  imageConceptFile = dataDir + 'flickr30k_no_NULL_top_100_trg.txt'
  if args.feat_type == 'synthetic':
    imageFeatureFile = dataDir + 'flickr30k_no_NULL_top_100_gaussian'
  elif args.feat_type == 'vgg16':
    imageFeatureFile = dataDir + 'flickr30k_no_NULL_top_100_vgg_penult.npz'
  
  conceptIdxFile = 'data/flickr30k/concept2idx_no_NULL_top_100.json'
  goldAlignmentFile = dataDir + 'flickr30k_no_NULL_top_100_gold_alignment.json'
  nWords = 100
else:
  raise ValueError('Dataset unspecified or invalid dataset')

modelConfigs = {
  'has_null': args.has_null, 
  'n_words': nWords, 
  'learning_rate': args.lr,
  'momentum': args.momentum, 
  'normalize_vfeat': args.normalize_vfeat, 
  'step_scale': args.step_scale, 
  'width': args.width,
  'hidden_dim': args.hidden_dim,
  'image_posterior_weights_file': args.image_posterior_weights_file
  }

if args.model_type == 'linear' or args.model_type == 'gaussian' or args.model_type == 'two-layer':
  if len(args.date) > 0:
    expDir = 'hmm_dnn/exp/%s_%s_%s_momentum%.1f_lr%.5f_stepscale%.2f_%s/' % (args.dataset, args.model_type, args.feat_type, modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale'], args.date) 
  else:
    expDir = 'hmm_dnn/exp/%s_%s_%s_momentum%.1f_lr%.5f_stepscale%.2f/' % (args.dataset, args.model_type, args.feat_type, modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale'])
elif args.model_type == 'clda':
  expDir = 'clda/exp/%s_%s_%s/' % (args.dataset, args.model_type, args.date)
else:
  raise ValueError('Model type not specified or invalid model type')

modelName = expDir + 'image_phone'
predAlignmentFile = modelName + '_alignment.json'

if not os.path.isdir(expDir):
  print('Create a new directory: ', expDir)
  os.mkdir(expDir)

modelName = expDir + 'image_phone'
print('Experiment directory: ', expDir)
   
# XXX
nReps = 5
SNRs = [40] #[40, 30, 20, 10, 5] 

tasks = [1, 2]
#-------------------------------#
# Feature extraction for MSCOCO #
#-------------------------------#
if 0 in tasks:    
  vCorpus = {}
  concept2idx = {}
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
  
  with open(goldAlignmentFile, 'r') as f:
    gold_info = json.load(f) 

  # Generate nTypes different clusters
  imgFeatDim = 2
  permute = False
  centroids = 10 * np.random.normal(size=(nTypes, imgFeatDim)) 
  
  for snr in SNRs:      
    for rep in range(nReps):
      noiseLevel = 10 * 10. ** (-snr / 20.) 
      for ex, (vSenStr, align_info) in enumerate(zip(vCorpusStr, gold_info)):
        N = len(vSenStr)
        alignment = align_info['alignment']

        if args.feat_type == 'synthetic':
          vSen = np.zeros((N, imgFeatDim))
          for i in range(N):
            vWord = vSenStr[i]
            vSen[i] = centroids[concept2idx[vWord]] + noiseLevel * np.random.normal(size=(imgFeatDim,)) 
                
        vCorpus['arr_'+str(ex)] = vSen

      if args.feat_type == 'synthetic':
        np.savez(imageFeatureFile + '_SNR_{}dB_{}.npz'.format(snr, rep), **vCorpus)
        with open(conceptIdxFile, 'w') as f:
          json.dump(concept2idx, f, indent=4, sort_keys=True)

#----------------#
# Model Training #
#----------------#
if 1 in tasks:
  print('Start training the model ...')
  #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
  begin_time = time.time()
  if args.feat_type == 'synthetic':
    for snr in SNRs:
      for rep in range(nReps):
        modelName = expDir + 'image_phone_{}dB_{}'.format(snr, rep)
        predAlignmentFile = modelName + '_alignment.json'
        if args.model_type == 'linear':
          model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile + '_SNR_{}dB_{}.npz'.format(snr, rep), modelConfigs, modelName=modelName)
        elif args.model_type == 'gaussian':
          model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile + '_SNR_{}dB_{}.npz'.format(snr, rep), modelConfigs, modelName=modelName)
        elif args.model_type == 'two-layer':
          model = ImagePhoneHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile + '_SNR_{}dB_{}.npz'.format(snr, rep), modelConfigs, modelName=modelName)
        model.trainUsingEM(25, writeModel=True, debug=False)
        model.printAlignment(modelName+'_alignment', debug=False)
    print('Take %.5s s to finish training and decoding the model !' % (time.time() - begin_time)) 
  else:
    if args.model_type == 'linear':
      model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    elif args.model_type == 'gaussian':
      model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    elif args.model_type == 'two-layer':
      model = ImagePhoneHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)

    model.trainUsingEM(20, writeModel=True, debug=False)
    print('Take %.5s s to finish training the model !' % (time.time() - begin_time))
    #model.simulatedAnnealing(numIterations=100, T0=1., debug=False)
    model.printAlignment(modelName+'_alignment', debug=False) 
    print('Take %.5s s to finish decoding !' % (time.time() - begin_time))

#------------#
# Evaluation #
#------------#
if 2 in tasks:
  with open(goldAlignmentFile, 'r') as f:
    gold_info = json.load(f)

  with open(conceptIdxFile, 'r') as f:
    concept2idx = json.load(f)

  if args.feat_type == 'synthetic': 
    for snr in SNRs:
      accs = []
      f1s = []
      purities = []
      for rep in range(nReps):
        modelName = expDir + 'image_phone_{}dB_{}'.format(snr, rep)
        print(modelName)
        predAlignmentFile = modelName + '_alignment.json'
        with open(predAlignmentFile, 'r') as f:
          pred_info = json.load(f)
        
        pred, gold = [], []
        for p, g in zip(pred_info, gold_info):
          pred.append(p['image_concepts'])
          if args.dataset == 'flickr':
            gold.append([concept2idx[c] for c in g['image_concepts']]) 
          elif args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
            gold.append(g['image_concepts'])
          else:
            raise ValueError('Invalid Dataset')
        purities.append(cluster_confusion_matrix(gold, pred, file_prefix='image_confusion_matrix', print_result=False, return_result=True))
        #cluster_confusion_matrix(gold, pred, alignment=gold_info, file_prefix='audio_confusion_matrix') 
        acc = accuracy(pred_info, gold_info)
        rec, prec, f1 = boundary_retrieval_metrics(pred_info, gold_info, return_results=True, print_results=False)
        accs.append(acc)
        f1s.append(f1)
      print('Average purities and deviation: ', np.mean(purities), np.var(purities)**.5)
      print('Average accuracy and deviation: ', np.mean(accs), np.var(accs)**.5)
      print('Average F1 score and deviation: ', np.mean(f1s), np.var(f1s)**.5)
      
  else:   
    with open(predAlignmentFile, 'r') as f:
      pred_info = json.load(f)
    
    pred, gold = [], []
    for p, g in zip(pred_info, gold_info):
      pred.append(p['image_concepts'])
      if args.dataset == 'flickr':
        gold.append([concept2idx[c] for c in g['image_concepts']]) 
      elif args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
        gold.append(g['image_concepts'])
      else:
        raise ValueError('Invalid Dataset')

    cluster_confusion_matrix(gold, pred, file_prefix='image_confusion_matrix')
    #cluster_confusion_matrix(gold, pred, alignment=gold_info, file_prefix='audio_confusion_matrix') 
    print('Alignment accuracy: ', accuracy(pred_info, gold_info))
    boundary_retrieval_metrics(pred_info, gold_info)

#---------------#
# Visualization #
#---------------#
if 3 in tasks:
  if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
    with open(dataDir + 'concept2idx.json', 'w') as f:
      json.dump({i:i for i in range(65)}, f, indent=4, sort_keys=True)
    
  f1_scores = plot_F1_score_histogram(predAlignmentFile, goldAlignmentFile, concept2idx_file=conceptIdxFile, draw_plot=True, out_file=modelName+'_f1_histogram') 
