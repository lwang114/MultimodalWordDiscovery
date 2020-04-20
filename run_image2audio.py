import json
import argparse
import shutil
import time
import os
import numpy as np
# from clda.image_audio_gmm_word_discoverer import ImageAudioGMMWordDiscoverer  
# from clda.image_audio_gmm_word_discoverer_fixed_var import ImageAudioGMMWordDiscovererFixedVar  
from hmm_dnn.image_audio_gaussian_hmm_word_discoverer import *
from hmm_dnn.image_audio_hmm_word_discoverer import *
from hmm_dnn.image_phone_gaussian_hmm_word_discoverer import *
from hmm_dnn.image_phone_hmm_word_discoverer import *
from hmm_dnn.image_phone_hmm_dnn_word_discoverer import *
from utils.clusteval import * 
from utils.postprocess import *

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--has_null', help='Include NULL symbol in the image feature', action='store_true')
parser.add_argument('--dataset', choices={'mscoco2k', 'mscoco20k', 'flickr'}, help='Dataset used for training the model')
parser.add_argument('--image_feat_type', choices={'synthetic', 'vgg16_penult', 'res34'}, help='Type of image features')
parser.add_argument('--audio_feat_type', choices={'synthetic', 'force_align', 'kamper', 'blstm_mean', 'blstm_last'}, help='Type of acoustic features')
parser.add_argument('--model_type', choices={'linear', 'gaussian', 'two-layer', 'clda'}, default='gaussian', help='Word discovery model type')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum used for GD iterations (hmm-dnn only)')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate used for GD iterations (hmm-dnn only)')
parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension (two-layer hmm-dnn only)')
parser.add_argument('--step_scale', type=float, default=0.1, help='Random jump step scale for simulated annealing (hmm-dnn only)')
parser.add_argument('--width', type=float, default=1., help='width parameter of the radial basis activation function (hmm-dnn only)')
parser.add_argument('--image_posterior_weights_file', type=str, default=None, help='Pretrained weights for the image posteriors')
parser.add_argument('--audio_posterior_weights_file', type=str, default=None, help='Pretrained weights for the audio posteriors')
parser.add_argument('--date', type=str, default='', help='Date of starting the experiment')
parser.add_argument('--n_phones', type=int, default=None, help='Number of phone-like clusters')

args = parser.parse_args()

if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
  dataDir = 'data/mscoco/'
  if args.dataset == 'mscoco2k':
    phoneCaptionFile = 'data/mscoco/mscoco2k_phone_captions.txt'
    goldAlignmentFile = dataDir + 'mscoco2k_gold_alignment.json'
  else:
    phoneCaptionFile = 'data/mscoco/mscoco20k_phone_captions.txt'
    goldAlignmentFile = dataDir + 'mscoco20k_gold_alignment.json'
  conceptIdxFile = dataDir + 'concept2idx.json'

  if args.audio_feat_type == 'synthetic':
    if args.dataset == 'mscoco2k':
      speechFeatureFile = dataDir + 'mscoco2k_subset_subword_level_phone_gaussian_vectors.npz'
    else:
      speechFeatureFile = dataDir + 'mscoco20k_phone_gaussian_vectors.npz'
  elif args.audio_feat_type == 'kamper': 
    if args.dataset == 'mscoco2k':
      speechFeatureFile = dataDir + 'mscoco_kamper_embeddings_phone_power_law.npz'
    else:
      speechFeatureFile = dataDir + 'mscoco20k_kamper_embeddings.npz'
  elif args.audio_feat_type == 'force_align':
    if args.dataset == 'mscoco2k':
      speechFeatureFile = dataDir + 'mscoco2k_force_align.txt'
    elif args.dataset == 'mscoco20k':
      speechFeatureFile = dataDir + 'mscoco20k_force_align.txt'

  elif args.audio_feat_type == 'blstm_mean': 
    if args.dataset == 'mscoco2k':
      speechFeatureFile = dataDir + 'mscoco_subset_2k_blstm_mean.npz'
    elif args.dataset == 'mscoco20k':
      speechFeatureFile = dataDir + 'mscoco20k_blstm_mean.npz'

  elif args.audio_feat_type == 'blstm_last':
    if args.dataset == 'mscoco2k': 
      speechFeatureFile = dataDir + 'mscoco_subset_2k_blstm_last.npz'
    elif args.dataset == 'mscoco20k':
      speechFeatureFile = dataDir + 'mscoco20k_blstm_last.npz'

  if args.image_feat_type == 'synthetic':
    if args.dataset == 'mscoco2k':
      imageFeatureFile = dataDir + 'mscoco2k_subset_subword_level_concept_gaussian_vectors.npz'
    else:
      imageFeatureFile = dataDir + 'mscoco20k_concept_gaussian_vectors.npz'

  elif args.image_feat_type == 'vgg16_penult':
    if args.dataset == 'mscoco2k':
      imageFeatureFile = dataDir + 'mscoco_vgg_penult.npz'
    else:
      imageFeatureFile = dataDir + 'mscoco_subset_130k_vgg16_penult.npz'

  elif args.image_feat_type == 'res34':
    if args.dataset == 'mscoco2k':
      imageFeatureFile = dataDir + 'mscoco_subset_2k_res34_embed512dim.npz'
    else:
      imageFeatureFile = dataDir + 'mscoco_subset_130k_res34_embed512dim.npz'

  imageConceptFile = dataDir + 'trg_mscoco_subset_subword_level_power_law.txt'
  nWords = 65
  if args.n_phones is None:
    if args.model_type == 'gaussian':
      nPhones = 42
    else:
      nPhones = 49
  else:
    nPhones = args.n_phones
elif args.dataset == 'flickr':
  goldAlignFile = dataDir + 'sensory_level/flickr30k_gold_alignment.json'
  speechFeatureFile = dataDir + 'sensory_level/flickr_concept_kamper_embeddings.npz'
  imageFeatureFile = dataDir + 'sensory_level/flickr30k_vgg_penult.npz'
  conceptIdxFile = dataDir + 'concept2idx_no_NULL_top_100.json'
  nWords = 100
  nPhones = 60
else:
  raise ValueError('Invalid dataset')

modelConfigs = {
  'has_null': args.has_null, 
  'n_words': nWords, 
  'n_phones': nPhones,
  'learning_rate': args.lr,
  'momentum': args.momentum, 
  'step_scale': args.step_scale, 
  'width': args.width,
  'hidden_dim': args.hidden_dim,
  'image_posterior_weights_file': args.image_posterior_weights_file,
  'audio_posterior_weights_file': args.audio_posterior_weights_file
  }

if args.model_type == 'linear' or args.model_type == 'gaussian' or args.model_type == 'two-layer':
  if len(args.date) > 0:
    expDir = 'hmm_dnn/exp/%s_%s_%s_%s_momentum%.1f_lr%.5f_stepscale%.2f_%s/' % (args.dataset, args.model_type, args.audio_feat_type, args.image_feat_type, modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale'], args.date) 
  else:
    expDir = 'hmm_dnn/exp/%s_%s_%s_%s_momentum%.1f_lr%.5f_stepscale%.2f/' % (args.dataset, args.model_type, args.audio_feat_type, args.image_feat_type, modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale'])
elif args.model_type == 'clda':
  expDir = 'clda/exp/%s_%s_%s/' % (args.dataset, args.model_type, args.date)
else:
  raise ValueError('Model type not specified or invalid model type')

if not os.path.isdir(expDir):
  os.mkdir(expDir)

modelName = expDir + 'image_audio'
predAlignmentFile = modelName + '_alignment.json'

if not os.path.isdir(expDir):
  print('Create a new directory: ', expDir)
  os.mkdir(expDir)

modelName = expDir + 'image_audio'
print('Experiment directory: ', expDir)

nReps = 5
SNRs = [40] 

tasks = [3]
if 0 in tasks:
  if args.dataset == 'mscoco2k':
    speechFeatureFile = 'data/mscoco/mscoco_subset_subword_level_phone_gaussian_vectors.npz'
    imageConceptFile = 'data/mscoco/trg_mscoco_subset_subword_level_power_law.txt'
    imageFeatureFile = 'data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
  elif args.dataset == 'mscoco20k':
    phoneCaptionFile = 'data/mscoco/mscoco20k_phone_captions.txt'
    speechFeatureFile = 'data/mscoco/mscoco20k_phone_gaussian_vectors.npz'
    imageConceptFile = 'data/mscoco/trg_mscoco_subset_130k_power_law_phone_captions.txt'
    imageFeatureFile = 'data/mscoco/mscoco20k_concept_gaussian_vectors.npz'
  
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
  
  # Generate nTypes different clusters
  imgFeatDim = 2
  centroids = 10 * np.random.normal(size=(nTypes, imgFeatDim)) 
   
  for ex, vSenStr in enumerate(vCorpusStr):
    N = len(vSenStr)     
    vSen = np.zeros((N, imgFeatDim))
    for i, vWord in enumerate(vSenStr):
      vSen[i] = centroids[concept2idx[vWord]] + 0.1 * np.random.normal(size=(imgFeatDim,))
    vCorpus['arr_'+str(ex)] = vSen
  
  np.savez(imageFeatureFile, **vCorpus)
  
  with open(conceptIdxFile, 'w') as f:
    json.dump(concept2idx, f, indent=4, sort_keys=True)
  
  aCorpus = {}
  phone2idx = {}
  nPhones = 0
  with open(phoneCaptionFile, 'r') as f:
    aCorpusStr = []
    for line in f:
      aSen = line.strip().split()
      aCorpusStr.append(aSen)
      for aWord in aSen:
        if aWord not in phone2idx:
          phone2idx[aWord] = nPhones
          nPhones += 1  
  print(nPhones)

  # Generate nPhones different clusters
  spFeatDim = 2
  centroids = 10 * np.random.normal(size=(nPhones, spFeatDim)) 
   
  for ex, aSenStr in enumerate(aCorpusStr):
    T = len(aSenStr)
    if args.audio_feat_type == 'one-hot':
      aSen = np.zeros((T, nPhones))
      for i, aWord in enumerate(aSenStr):
        aSen[i, phone2idx[aWord]] = 1.
    elif args.audio_feat_type == 'gaussian':
      aSen = np.zeros((T, spFeatDim))
      for i, aWord in enumerate(aSenStr):
        aSen[i] = centroids[phone2idx[aWord]] + 0.1 * np.random.normal(size=(spFeatDim,))
    aCorpus['arr_'+str(ex)] = aSen
  
  np.savez(speechFeatureFile, **aCorpus)

if 1 in tasks:
  print('Start training ...')
  begin_time = time.time()
  
  if args.audio_feat_type == 'synthetic' and args.image_feat_type == 'synthetic':
    for snr in SNRs:
      for rep in range(nReps):
        modelName = expDir + 'image_phone_{}dB_{}'.format(snr, rep)
        predAlignmentFile = modelName + '_alignment.json'
        if args.model_type == 'linear':
          model = ImageAudioHMMWordDiscoverer(speechFeatureFile, imageFeatureFile + '_SNR_{}dB_{}.npz'.format(snr, rep), modelConfigs, modelName=modelName)
        elif args.model_type == 'gaussian':
          model = ImageAudioGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile + '_SNR_{}dB_{}.npz'.format(snr, rep), modelConfigs, modelName=modelName)
        elif args.model_type == 'two-layer':
          model = ImageAudioHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile + '_SNR_{}dB_{}.npz'.format(snr, rep), modelConfigs, modelName=modelName)
        model.trainUsingEM(25, writeModel=True, debug=False)
        model.printAlignment(modelName+'_alignment', debug=False)
    print('Take %.5s s to finish training and decoding the model !' % (time.time() - begin_time)) 
  elif args.audio_feat_type == 'force_align':
    if args.model_type == 'linear':
      model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    elif args.model_type == 'gaussian':
      model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    elif args.model_type == 'two-layer':
      model = ImagePhoneHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=False, debug=False)
    print('Take %.5s s to finish training the model !' % (time.time() - begin_time))
    #model.simulatedAnnealing(numIterations=100, T0=1., debug=False)
    model.printAlignment(modelName+'_alignment', debug=False) 
    print('Take %.5s s to finish decoding !' % (time.time() - begin_time)) 
  else:
    if args.model_type == 'linear':
      model = ImageAudioHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    elif args.model_type == 'gaussian':
      model = ImageAudioGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    elif args.model_type == 'two-layer':
      model = ImageAudioHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)

    model.trainUsingEM(20, writeModel=False, debug=False)
    print('Take %.5s s to finish training the model !' % (time.time() - begin_time))
    #model.simulatedAnnealing(numIterations=100, T0=1., debug=False)
    model.printAlignment(modelName+'_alignment', debug=False) 
    print('Take %.5s s to finish decoding !' % (time.time() - begin_time))
  
if 2 in tasks:
  with open(goldAlignmentFile, 'r') as f:
    gold_info = json.load(f)

  with open(conceptIdxFile, 'r') as f:
    concept2idx = json.load(f)
  
  if args.audio_feat_type == 'synthetic' and args.image_feat_type: 
    for snr in SNRs:
      accs = []
      f1s = []
      vpurities = []
      apurities = []
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
        vpurities.append(cluster_confusion_matrix(pred, gold, file_prefix='image_confusion_matrix', print_result=False, return_result=True))  
        if args.dataset == 'flickr':
          apurities.append(word_cluster_confusion_matrix(pred_info, gold_info, concept2idx=concept2idx, file_prefix='audio_confusion_matrix'))
        else:
          apurities.append(word_cluster_confusion_matrix(pred_info, gold_info, file_prefix='audio_confusion_matrix'))
        acc = accuracy(pred_info, gold_info)
        
        rec, prec, f1 = boundary_retrieval_metrics(pred_info, gold_info, return_results=True, print_results=False)
        accs.append(acc)
        f1s.append(f1)
      print('Average word cluster purities and deviation: ', np.mean(apurities), np.var(apurities)**.5)
      print('Average visual concept cluster purities and deviation: ', np.mean(vpurities), np.var(vpurities)**.5)
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
    print('Concept cluster purity:')
    cluster_confusion_matrix(pred, gold, file_prefix='image_confusion_matrix') 

    print('Word cluster purity:')
    if args.dataset == 'flickr':
      word_cluster_confusion_matrix(pred_info, gold_info, concept2idx=concept2idx, file_prefix='audio_confusion_matrix') 
    else:
      word_cluster_confusion_matrix(pred_info, gold_info, file_prefix='audio_confusion_matrix') 

    print('Alignment accuracy: ', accuracy(pred_info, gold_info))
    boundary_retrieval_metrics(pred_info, gold_info)

if 3 in tasks:
  start_time = time.time()
  filePrefix = expDir + '_'.join(['image2audio', args.dataset, args.model_type, args.audio_feat_type, args.image_feat_type])
  alignment_to_word_classes(predAlignmentFile, phoneCaptionFile, word_class_file='_'.join([filePrefix, 'words.class']))
  alignment_to_word_units(predAlignmentFile, phoneCaptionFile, word_unit_file='_'.join([filePrefix, 'word_units.wrd']), phone_unit_file='_'.join([filePrefix, 'phone_units.phn']), include_null=True) 
# TODO: visualizations
# print("Generating plots ...")
