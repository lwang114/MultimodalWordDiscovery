import pkg_resources 
from WDE.readers.gold_reader import *
from WDE.readers.disc_reader import *
from WDE.measures.grouping import * 
from WDE.measures.coverage import *
from WDE.measures.boundary import *
from WDE.measures.ned import *
from WDE.measures.token_type import *
from postprocess import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default='./', help='Experimental directory containing the alignment files')
parser.add_argument('--dataset', choices=['mscoco2k', 'mscoco20k', 'flickr'])
args = parser.parse_args()

tasks = [1]
#--------------------------#
# Extract Discovered Words #
#--------------------------#
if 0 in tasks:
  if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
    datapath = '../data/mscoco/'
    phone_corpus = datapath + '%s_phone_captions.txt' % args.dataset
    concept_corpus = datapath + '%s_image_captions.txt' % args.dataset
    concept2id_file = datapath + 'concept2idx.json'
    gold_alignment_file = datapath + '%s_gold_alignment.json' % args.dataset
    with open(args.exp_dir+'model_names.txt', 'r') as f:
      model_names = f.read().strip().split()
    pred_alignment_files = ['%s/%s_%s_pred_alignment.json' % (args.exp_dir, args.dataset, model_name) for model_name in model_names]
  elif args.dataset == 'flickr':
    datapath = '../data/flickr30k/'
    phone_corpus = datapath + 'phoneme_level/src_flickr30k.txt'
    concept_corpus = datapath + 'phoneme_level/trg_flickr30k.txt'
    concept2id_file = None
    gold_alignment_file = datapath + 'phoneme_level/%s_gold_alignment.json' % args.dataset
    # TODO Make this more general 
    model_names = ['mixture', 'hmm', 'nmt-novt', 'nmt-novc']
    pred_alignment_files = ['../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_alignment.json', '../hmm/exp/aug_31_flickr/flickr30k_pred_alignment.json', '../nmt/exp/feb26_normalize_over_time/output/alignment.json', '../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json']
    
  alignment_to_word_units(gold_alignment_file, phone_corpus, concept_corpus, word_unit_file='tdev2/WDE/share/%s_word_units.wrd' % args.dataset, phone_unit_file='tdev2/WDE/share/%s_phone_units.phn' % args.dataset, include_null=True, concept2id_file=concept2id_file)
  for i, (model_name, pred_alignment_file) in enumerate(zip(model_names, pred_alignment_files)):
    discovered_word_file = 'tdev2/WDE/share/discovered_words_%s.class' % model_name
    alignment_to_word_classes(pred_alignment_file, phone_corpus, word_class_file=discovered_word_file, include_null=True)

#------------------------#
# Phone-level Evaluation #
#------------------------#
if 1 in tasks:
  wrd_path = pkg_resources.resource_filename(
              pkg_resources.Requirement.parse('WDE'),
                          'WDE/share/%s_word_units.wrd' % args.dataset)
  phn_path = pkg_resources.resource_filename(
              pkg_resources.Requirement.parse('WDE'),
                          'WDE/share/%s_phone_units.phn' % args.dataset)

  gold = Gold(wrd_path=wrd_path, 
                phn_path=phn_path) 
  
  if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
    with open(args.exp_dir+'model_names.txt', 'r') as f:
      model_names = f.read().strip().split()
  else:
    model_names = ['enriched'] #['nmt-novt', 'nmt-novc']
  disc_clsfiles = ['tdev2/WDE/share/discovered_words_%s.class' % model_name for model_name in model_names]

  for model_name, disc_clsfile in zip(model_names, disc_clsfiles):
    discovered = Disc(disc_clsfile, gold) 
    
    print(model_name)
    grouping = Grouping(discovered)
    grouping.compute_grouping()
    print('Grouping precision and recall: ', grouping.precision, grouping.recall)
    #print('Grouping fscore: ', grouping.fscore)

    coverage = Coverage(gold, discovered)
    coverage.compute_coverage()
    print('Coverage: ', coverage.coverage)

    boundary = Boundary(gold, discovered)
    boundary.compute_boundary()
    print('Boundary precision and recall: ', boundary.precision, boundary.recall)
    #print('Boundary fscore: ', boundary.fscore)

    ned = Ned(discovered)
    ned.compute_ned()
    print('NED: ', ned.ned)

    token_type = TokenType(gold, discovered)
    token_type.compute_token_type()
    print('Token type precision and recall: ', token_type.precision, token_type.recall)
    #print('Token type fscore: ', token_type.fscore)

    with open('%s_scores.txt' % model_name, 'w') as f:
      f.write('Grouping precision: %.5f, recall: %.5f\n' % (grouping.precision, grouping.recall))
      f.write('Boundary precision: %.5f, recall: %.5f\n' % (boundary.precision, boundary.recall))
      f.write('Token/type precision: %.5f %.5f, recall: %.5f %.5f\n' % (token_type.precision[0], token_type.precision[1], token_type.recall[0], token_type.recall[1]))
      f.write('Coverage: %.5f\n' % coverage.coverage)
      f.write('ned: %.5f\n' % ned.ned)

# TODO
#------------------------#
# Audio-level Evaluation #
#------------------------#
if 2 in tasks:
  wrd_path = pkg_resources.resource_filename(
              pkg_resources.Requirement.parse('WDE'),
                          'WDE/share/flickr30k_audio_word_units.wrd')
  phn_path = pkg_resources.resource_filename(
              pkg_resources.Requirement.parse('WDE'),
                          'WDE/share/flickr30k_audio_phone_units.phn')

  gold = Gold(wrd_path=wrd_path, 
                  phn_path=phn_path) 

  model_names = ['seg_gmm_bn', 'seg_hmm_bn', 'seg_gmm_mfcc', 'seg_hmm_mfcc', 'gmm_bn', 'kmeans_bn', 'gmm_mfcc', 'kmeans_mfcc']
  disc_clsfiles = ['tdev2/WDE/share/discovered_words_%s.class' % model_name for model_name in model_names]
  for model_name, disc_clsfile in zip(model_names, disc_clsfiles):
    print(model_name)
    discovered = Disc(disc_clsfile, gold)
    
    grouping = Grouping(discovered)
    grouping.compute_grouping()
    print('Grouping precision and recall: ', grouping.precision, grouping.recall)
    #print('Grouping fscore: ', grouping.fscore)

    coverage = Coverage(gold, discovered)
    coverage.compute_coverage()
    print('Coverage: ', coverage.coverage)

    boundary = Boundary(gold, discovered)
    boundary.compute_boundary()
    print('Boundary precision and recall: ', boundary.precision, boundary.recall)
    #print('Boundary fscore: ', boundary.fscore)

    ned = Ned(discovered)
    ned.compute_ned()
    print('NED: ', ned.ned)

    token_type = TokenType(gold, discovered)
    token_type.compute_token_type()
    print('Token type precision and recall: ', token_type.precision, token_type.recall)
    #print('Token type fscore: ', token_type.fscore)

    with open('%s_scores.txt' % model_name, 'w') as f:
      f.write('Grouping precision: %.5f, recall: %.5f\n' % (grouping.precision, grouping.recall))
      f.write('Boundary precision: %.5f, recall: %.5f\n' % (boundary.precision, boundary.recall))
      f.write('Token/type precision: %.5f %.5f, recall: %.5f %.5f\n' % (token_type.precision[0], token_type.precision[1], token_type.recall[0], token_type.recall[1]))
      f.write('Coverage: %.5f\n' % coverage.coverage)
      f.write('ned: %.5f\n' % ned.ned)    
