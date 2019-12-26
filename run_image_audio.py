import json
import argparse
import shutil
import time
import os
import numpy as np
from hmm.image_audio_gmm_word_discoverer import ImageAudioGMMWordDiscoverer  
from hmm.image_audio_gmm_word_discoverer_fixed_var import ImageAudioGMMWordDiscovererFixedVar  
from utils.clusteval import * 

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default='./', help='Experiment directory with / at the end')
parser.add_argument('--dataset', choices={'mscoco', 'flickr'}, default='mscoco')
parser.add_argument('--data_path', type=str, default='data/mscoco/', help='Data directory with / at the end')
parser.add_argument('--num_mixtures', type=int, default=1, help='Number of mixtures for GMM')
parser.add_argument('--num_concepts', type=int, default=100, help='Number of mixtures for GMM')
parser.add_argument('--embedding_dim', type=int, default=140, help='Acoustic embedding dimension')
parser.add_argument('--gamma_sb0', type=float, default=1., help='Sparsity parameter for the stick breaking process')
parser.add_argument('--curriculum_learning', action='store_true', help='Use curriculum learning')
parser.add_argument('--fixed_var', action='store_true', help='Fixed the variance parameters of model')
args = parser.parse_args()

if args.dataset == 'mscoco':
  if args.curriculum_learning:
    gold_align_file = args.data_path + 'mscoco_gold_alignment_syllabus_1.json'
    speech_feature_file = args.data_path + 'mscoco_kamper_embeddings_syllabus_1.npz'
    image_feature_file = args.data_path + 'mscoco_vgg_penult_syllabus_1.npz'
  else:
    gold_align_file = args.data_path + 'mscoco_gold_alignment.json'
    speech_feature_file = args.data_path + 'mscoco_kamper_embeddings.npz'
    image_feature_file = args.data_path + 'mscoco_vgg_penult.npz'
  syllabus_files = [(args.data_path + 'mscoco_kamper_embeddings_syllabus_%d.npz' % i_s, args.data_path + 'mscoco_vgg_penult_syllabus_%d.npz' % i_s) for i_s in range(2)] 
else:
  gold_align_file = args.data_path + 'sensory_level/flickr30k_gold_alignment.json'
  speech_feature_file = args.data_path + 'sensory_level/flickr_concept_kamper_embeddings.npz'
  image_feature_file = args.data_path + 'sensory_level/flickr30k_vgg_penult.npz'

pred_align_file = args.exp_dir + 'predict_alignment.json'
if not os.path.isdir(args.exp_dir):
  os.mkdir(args.exp_dir)

start = 0
end = 3

if start < 1 and end >= 1:
  print('Start training ...')
  begin_time = time.time()
  model_configs = {'Kmax':args.num_concepts, 'Mmax':args.num_mixtures, 'embedding_dim':args.embedding_dim, 'has_null':False}  
  
  # TODO: Merge the two cases
  if args.fixed_var:
    model = ImageAudioGMMWordDiscovererFixedVar(speech_feature_file, image_feature_file, model_configs=model_configs, model_name=args.exp_dir+'image_audio')
  else:
    model = ImageAudioGMMWordDiscoverer(speech_feature_file, image_feature_file, model_configs=model_configs, model_name=args.exp_dir+'image_audio')
  
  if args.curriculum_learning:
    model.train_curriculum_EM(syllabus_files, num_iterations=20)
  else:
    model.train_using_EM(num_iterations=20)
  print('Finish training after %f s !' % (time.time() - begin_time))

if start < 2 and end >= 2:
  print("Start evaluation ...")
  begin_time = time.time()

  if args.dataset == 'mscoco' and not os.path.isfile(gold_align_file):
    if args.curriculum_learning:
      data_info_file = args.data_path + 'mscoco_subset_concept_info_syllabus_1.json'
      concept_info_file = args.data_path + 'mscoco_subset_concept_counts_syllabus_1.json'
    else:
      data_info_file = args.data_path + 'mscoco_subset_concept_info_power_law.json'
      concept_info_file = args.data_path + 'mscoco_subset_concept_counts_power_law.json'


    gold_info = []
    with open(data_info_file, 'r') as f:
      data_info = json.load(f)
    with open(concept_info_file, 'r') as f:
      concept_counts = json.load(f)
      concept_names = sorted(concept_counts)
      name2int = {c: i for i, c in enumerate(concept_names)} 

    for i, data_id in enumerate(sorted(data_info, key=lambda x:int(x.split('_')[-1]))):
      l = len(data_info[data_id]['concepts'])
      concept_names = data_info[data_id]['concepts']
      concepts = [name2int[name] for name in concept_names]
      g_info = {
                'image_concepts': concepts,
                'image_concept_names': concept_names,
                'alignment': np.arange(l).tolist(),
                'index': i
              }
      gold_info.append(g_info)

    with open(gold_align_file, 'w') as f:
      json.dump(gold_info, f, indent=4, sort_keys=True)
  
  # XXX: make this more general
  pred_align_file = args.exp_dir + 'image_audio.json'
  with open(pred_align_file, 'r') as f:
    pred_aligns = json.load(f)

  with open(gold_align_file, 'r') as f:
    gold_aligns = json.load(f)

  pred_concepts = [pred['image_concepts'] for pred in pred_aligns]
  gold_concepts = [gold['image_concepts'] for gold in gold_aligns]
  print('Accuracy: ', accuracy(pred_aligns, gold_aligns))
  boundary_retrieval_metrics(pred_aligns, gold_aligns)
  cluster_confusion_matrix(pred_concepts, gold_concepts, file_prefix=args.exp_dir+'image_confusion_matrix')
  cluster_confusion_matrix(pred_concepts, gold_concepts, alignment=pred_aligns, file_prefix=args.exp_dir+'audio_confusion_matrix')
    
  print('Finish evaluation after %f s !' % (time.time() - begin_time))

# TODO: visualizations
if start < 3 and end >= 3:
  start_time = time.time()
  print("Generating plots ...")
