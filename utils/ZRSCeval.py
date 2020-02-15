import pkg_resources 
from WDE.readers.gold_reader import *
from WDE.readers.disc_reader import *
from WDE.measures.grouping import * 
from WDE.measures.coverage import *
from WDE.measures.boundary import *
from WDE.measures.ned import *
from WDE.measures.token_type import *
from preprocess import *

# XXX
tasks = [2]
#--------------------------#
# Extract Discovered Words #
#--------------------------#
if 0 in tasks:
  datapath = '../data/' 
  instance_file = datapath + 'flickr30k/bboxes.mat'
  alignment_file = datapath + 'flickr30k/flickr30k_phrases.txt'
  caption_file = datapath + 'flickr30k/results_20130124.token'
  phone_corpus = datapath + 'flickr30k/phoneme_level/src_flickr30k.txt'
  category_file = 'vgg16_model/imagenet_class_index.json'

  preproc = Flickr_Preprocessor(instance_file, alignment_file, caption_file, category_file=category_file, image_path='../../data/flickr30k/flickr30k-images/')
  model_names = ['mixture', 'hmm', 'nmt-novt', 'nmt-novc']
  pred_alignment_files = ['../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_alignment.json', '../hmm/exp/aug_31_flickr/flickr30k_pred_alignment.json', '../nmt/exp/feb26_normalize_over_time/output/alignment.json', '../nmt/exp/feb28_phoneme_level_clustering/output/alignment.json']
    
  for i, (model_name, pred_alignment_file) in enumerate(zip(model_names, pred_alignment_files)):
    discovered_word_file = 'tdev2/WDE/share/discovered_words_%s.class' % model_name
    preproc.alignment_to_word_classes(pred_alignment_file, phone_corpus, discovered_word_file)

#------------------------#
# Phone-level Evaluation #
#------------------------#
if 1 in tasks:
  wrd_path = pkg_resources.resource_filename(
              pkg_resources.Requirement.parse('WDE'),
                          'WDE/share/flickr30k_word_units.wrd')
  phn_path = pkg_resources.resource_filename(
              pkg_resources.Requirement.parse('WDE'),
                          'WDE/share/flickr30k_phone_units.phn')

  gold = Gold(wrd_path=wrd_path, 
                phn_path=phn_path) 

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
