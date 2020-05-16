import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer 
import numpy as np
from scipy.io import loadmat
from PIL import Image
import time
import os

DEBUG = True
NULL = 'NULL'
PUNCT = [',', '\'', '\"', '/', '?', '>', '<', '#', '%', '&', '*']
NONVISUAL = 'notvisual'

class Flickr_Preprocessor(object):
  def __init__(self, instance_file, phrase_type_file, phone_segment_file, concept_class_file, word_segment_dir=None):
    self.instance_file = instance_file
    self.phrase_type_file = phrase_type_file
    self.phone_segment_file = phone_segment_file
    self.word_segment_dir = word_segment_dir
    self.lemmatizer = WordNetLemmatizer()
    self.ic = wordnet_ic.ic('ic-semcor.dat')
    self.concept_class_file = concept_class_file
    with open(self.concept_class_file, 'r') as f:
      self.concept_names = f.read().strip().split('\n')
      self.concept2ids = {c:i for i, c in enumerate(self.concept_names)}
    
    with open('flickr_type2idx.json', 'w') as f:
      json.dump(self.concept2ids, f, indent=4, sort_keys=True)

    self.word2concept = {}
    self.concept2word = {c:[] for c in self.concept_names}

  # TODO Process the phone segment file 
  # TODO Add phrase location
  # TODO Add phrase type
  def extract_info(self, out_file='flickr30k_info.json', word2concept_file=None, split_file=None):
    pairs = []
    if not word2concept_file:
      self.extract_word_to_concept_map()
    else:
      with open(word2concept_file, 'r') as f:
        self.word2concept = json.load(f)

    f_ty = open(self.phrase_type_file, 'r')
    phrase_types = f_ty.read().strip().split('\n')
    print(len(phrase_types))
    f_ty.close()
    if split_file:
      f_split = open(split_file, 'r')
      test_files = f_split.read().strip().split('\n')
      test_ids = [test_file.split('/')[-1].split('_')[0] for test_file in test_files]
      print(test_ids[:10])
    else:
      test_ids = []

    f_ins = open(self.instance_file, 'r')
    cur_capt_id = ''
    i = -1
    i_ex = 0
    for line in f_ins:
      parts = line.split()
      img_id = parts[0].split('_')[0]
      capt_id = parts[0]
      # XXX
      # if i < 24935 * 5:
      #   i += 1
      # continue 
      i += 1

      phrase = parts[1:-4]
      bbox = parts[-4:]
      phones = []
      for word in phrase:
        g2p_result = os.popen('phonetisaurus-g2pfst --model=g2ps/models/english_4_2_2.fst --word=%s' % word).read().strip().split()[2:]
        # with open('g2p_out.txt', 'r') as f_g2p:
        #   g2p_result = f_g2p.read().strip().split()[2:]
          
        phones += [ph if ph[0] != '\u02c8' else ph[1:] for ph in g2p_result]

      if capt_id != cur_capt_id:
        # Add train/test split information
        if img_id in test_ids:
          is_train = False
        else:
          is_train = True
        print(i, i_ex, is_train, phrase_types[i])
        pairs.append({
                  'index': i_ex,
                  'image_id': capt_id.split('.')[0].split('_')[0],
                  'image_filename:': capt_id.split('.')[0] + '.jpg',
                  'capt_id': capt_id, 
                  'phrases': []
                  'bbox': [],
                  'is_train': is_train,
                  'image_concepts': []
                 })
        i_ex += 1
        cur_capt_id = capt_id

      pairs[i_ex-1]['phrases'].append([phrase, phones])
      pairs[i_ex-1]['bbox'].append(bbox)
      pairs[i_ex-1]['image_concepts'].append(phrase_types[i].split()[1])
      
    f_ins.close()
    
    with open(out_file, 'w') as f:
      json.dump(pairs, f, indent=4, sort_keys=True)

  def extract_word_to_concept_map(self, configs={}):
    min_freq = configs.get('min_freq', 10)
    # Options: Leacock-Chodorow Similarity (lch) or Wu-Palmer Similarity (wup) or Lin Similarity
    sim_type = configs.get('sim_type', 'wup+res')
    if sim_type == 'jcn':
      sim_thres = 0.2
    elif sim_type == 'lin':
      sim_thres = 0.4
    elif sim_type == 'lch':
      sim_thres = 2.
    elif sim_type == 'wup':
      sim_thres = 0.5
    elif sim_type == 'wup+res':
      sim_thres = 1.0
    else:
      raise ValueError('Invalid similarity type')

    f_ins = open(self.instance_file, 'r')
    noun_counts = {}
    nouns = []
    # i = 0
    for line in f_ins:
      # XXX 
      # if i >= 20:
      #   break
      # i += 1
      parts = line.strip().split()
      img_id = parts[0]
      phrase = parts[1:-4]
      pos = nltk.pos_tag(phrase)[::-1]
      print(img_id)
      for w in pos:
        if w[1][0] == 'N':
          word = self.lemmatizer.lemmatize(w[0]).lower()
          nouns.append(word)
          if word not in noun_counts:
            noun_counts[word] = 1
          else:
            noun_counts[word] += 1
    f_ins.close()
    print('Number of nouns: ', len(noun_counts))
    
    begin_time = time.time()
    failed_words = []
    word_sub = None
    for word in nouns:
      if word == 'kayaker' or word == 'canoer' or word == 'kayakers' or word == 'canoers':
        word_sub = 'sailor'
      elif word == 'people' or word == 'group':
        word_sub = 'person'
      else:
        word_sub = word

      if noun_counts[word] < min_freq:
        continue
      elif word in self.word2concept:
        continue
      else:
        try:
          word_sense = wn.synset('%s.n.01' % word_sub)
          print(word_sense)
          word_senses = [word_sense]
        except:
          continue

        S = np.zeros((len(self.concept_names),))
        for i_c, c in enumerate(self.concept_names):
          concept_senses = [wn.synset('%s.n.01' % c)]
          S[i_c] = compute_word_similarity(concept_senses, word_senses, sim_type=sim_type, ic=self.ic)
        c_best = self.concept_names[np.argmax(S)]

        if np.max(S) < sim_thres or c_best == 'person':
          if word not in self.word2concept:
            print('Create a new concept: ', word, np.max(S), c_best)
            print('Number of concepts: ', len(self.concept_names))
            self.concept2word[word_sub] = [word]
            self.word2concept[word] = word_sub
            self.concept_names.append(word_sub)
          continue
        elif word not in self.word2concept:
          print('Add a new word: ', word, np.max(S), c_best)
          print('Number of nouns: ', len(self.word2concept))
          self.concept2word[c_best].append(word)
          self.word2concept[word] = c_best

    print('Take %s s to finish extracting concepts' % (time.time() - begin_time)) 
    with open('concept2word.json', 'w') as f:
      json.dump(self.concept2word, f, indent=4, sort_keys=True)

    with open('word2concept.json', 'w') as f:
      json.dump(self.word2concept, f, indent=4, sort_keys=True)
    print('Number of nouns: ', len(self.word2concept))
    print('Number of concepts: ', len(self.concept_names))
    
    with open('failed_words.txt', 'w') as f:
      f.write('\n'.join(failed_words))

  def create_gold_alignment(self, data_file, out_file='flickr30k_gold_alignment.json'):    
    with open(data_file, 'r') as f:
      data_info = json.load(f)

    align_info = []
    for i, datum_info in enumerate(data_info):
      phrases = datum_info['phrases']
      img_concepts = datum_info['image_concepts']
      phone_alignment = []
      word_alignment = []
      phone_segmentation = []
      word_segmentation = []
      caption = ''
      phone_start = 0
      word_start = 0
      for i_phr, phrase in enumerate(phrases):
        phone_alignment += [i_phr] * len(phrase[1])
        phone_segmentation.append([phone_start, phone_start+len(phrase[1])])
        phone_start = phone_start + len(phrase[1])

        word_alignment += [i_phr] * len(phrase[0])
        word_segmentation.append([word_start, word_start+len(phrase[0])])
        word_start = word_start + len(phrase[0])

        caption += ',' + ' '.join(phrase[0])
      align_info.append({
                         'index': i,
                         'is_train': datum_info['is_train'],
                         'phone_alignment': phone_alignment,
                         'word_alignment': word_alignment, 
                         'caption': caption, 
                         'image_concepts': img_concepts,
                         'phone_segmentation': phone_segmentation,
                         'word_segmentation': word_segmentation
                         })

    with open(out_file, 'w') as f:
      json.dump(align_info, f, indent=4, sort_keys=True)

  def create_captions(self, data_file, out_file='flickr30k_captions', split=False):
    word_captions = []
    phone_captions = [] 
    with open(data_file, 'r') as f:
      data_info = json.load(f)

    if split:
      with open(out_file+'_words_train.txt', 'w') as fwtr,\
           open(out_file+'_phones_train.txt', 'w') as fptr,\
           open(out_file+'_words_test.txt', 'w') as fwtx,\
           open(out_file+'_phones_test.txt', 'w') as fptx:
        for i, datum_info in enumerate(data_info):
          phrases = datum_info['phrases']
          img_concepts = datum_info['image_concepts']
          # print(len(phrases), len(img_concepts))
          # print(phrases)
          # print(img_concepts)
          phrases = [phr for i_phr, phr in enumerate(phrases) if img_concepts[i_phr] != NONVISUAL]
          is_train = datum_info['is_train']
          word_caption = []
          phone_caption = []
          for i_phr, phrase in enumerate(phrases):
            word_caption += phrase[0]
            phone_caption += phrase[1]
          if is_train:
            fwtr.write(' '.join(word_caption)+'\n')
            fptr.write(' '.join(phone_caption)+'\n') 
          else:
            fwtx.write(' '.join(word_caption)+'\n')
            fptx.write(' '.join(phone_caption)+'\n') 
    else:
      with open(out_file+'_words.txt', 'w') as fw,\
           open(out_file+'_phones.txt', 'w') as fp:
        for i, datum_info in enumerate(data_info):
          phrases = datum_info['phrases']
          img_concepts = datum_info['image_concepts']
          # print(len(phrases), len(img_concepts))
          # print(phrases)
          # print(img_concepts)
          phrases = [phr for i_phr, phr in enumerate(phrases) if img_concepts[i_phr] != NONVISUAL]
          is_train = datum_info['is_train']
          word_caption = []
          phone_caption = []
          for i_phr, phrase in enumerate(phrases):
            word_caption += phrase[0]
            phone_caption += phrase[1]
        
          fw.write(' '.join(word_caption)+'\n')
          fp.write(' '.join(phone_caption)+'\n')   
  
  def train_test_split(self, split_file, out_file='flickr30k_phrase'):
    if split_file:
      f_split = open(split_file, 'r')
      test_files = f_split.read().strip().split('\n')
      test_ids = [test_file.split('/')[-1].split('_')[0] for test_file in test_files]
    else:
      test_ids = []

    f_ins = open(self.instance_file, 'r')
    cur_capt_id = ''
    i = 0
    with open(out_file+'_bboxes_train.txt', 'w') as ftr,\
         open(out_file+'_bboxes_test.txt', 'w') as ftx:
      for line in f_ins:
        # XXX
        # if i > 10:
        #   break
        # i += 1
        
        parts = line.strip().split()
        img_id = parts[0].split('_')[0]
        capt_id = parts[0]
        print(capt_id, parts[-1])
         
        if img_id in test_ids:
          ftx.write(line)
        else:
          ftr.write(line)
    
    f_ty = open(self.phrase_type_file, 'r') 
    i = 0
    with open(out_file+'_types_train.txt', 'w') as ftr,\
         open(out_file+'_types_test.txt', 'w') as ftx:
      for line in f_ty:
        # XXX
        # if i > 10:
        #   break
        # i += 1
        
        parts = line.split()
        img_id = parts[0].split('_')[0]
        capt_id = parts[0]
        print(capt_id)
         
        if img_id in test_ids:
          ftx.write(line)
        else:
          ftr.write(line) 
  
  def cleanup_datafile(self, data_file, out_file='flickr30k_info.json'):
    with open(data_file, 'r') as f:
      data_info = json.load(f)

    new_data_info = []
    prev_capt_id = ''
    prev_img_file = ''
    prev_img_id = ''

    # XXX
    for i, datum in enumerate(data_info):
      if i == 0:
        new_datum = {
          'bbox': datum['bbox'],
          'phrases': datum['phrases'],
          'capt_id': '2571096893_694ce79768.jpg_1',
          'image_concepts': datum['image_concepts'],
          'image_filename': '2571096893_694ce79768.jpg',
          'image_id': '2571096893',
          'index': 0,
          'is_train': True,
          } 
        new_data_info.append(new_datum)
      else:
        new_datum = {
          'bbox': datum['bbox'],
          'phrases': [[datum['phrases'][0], datum['phrases'][1]]] + datum['phrases'][2:],
          'capt_id': prev_capt_id,
          'image_concepts': datum['image_concepts'],
          'image_filename': prev_img_file,
          'image_id': prev_img_id,
          'index': i,
          'is_train': datum['is_train']
          }
        new_data_info.append(new_datum)
      prev_capt_id = datum['capt_id']
      prev_img_file = datum['image_filename:']
      prev_img_id = datum['image_id']

    with open(out_file, 'w') as f:
      json.dump(new_data_info, f, indent=4, sort_keys=True)

def compute_word_similarity(word_senses1, word_senses2, sim_type='wup+res', pos='n', ic=None):
  scores = []
  n_senses = 0
  for s1 in word_senses1:
    for s2 in word_senses2:
      if s1.pos() == pos and s2.pos() == pos:
        n_senses += 1
        if sim_type == 'lch':
          scores.append(s1.lch_similarity(s2))
        elif sim_type == 'wup+res':
          if not ic:
            raise ValueError('Please provide an information content (IC) to compute Lin similarity')
          scores.append(s1.wup_similarity(s2) / s1.wup_similarity(s1) + s1.res_similarity(s2, ic) / s1.res_similarity(s1, ic))
        elif sim_type == 'wup':
          scores.append(s1.wup_similarity(s2))
        elif sim_type == 'lin':
          if not ic:
            raise ValueError('Please provide an information content (IC) to compute Lin similarity')
          scores.append(s1.lin_similarity(s2, ic))
        elif sim_type == 'jcn':
          if not ic:
            raise ValueError('Please provide an information content (IC) to compute Lin similarity')
          scores.append(s1.jcn_similarity(s2, ic)) 
        else:
          raise NotImplementedError
  return max(scores)

if __name__ == '__main__':
  tasks = [3] 
  preproc = Flickr_Preprocessor('../data/flickr30k/flickr30k_phrases_bboxes.txt', '../data/flickr30k/flickr30k_phrase_types.txt', None, concept_class_file='../data/flickr30k/flickr_classnames_original.txt')
  data_file = '../data/flickr30k/flickr30k_info.json'

  if 0 in tasks:
    preproc.extract_word_to_concept_map()
  if 1 in tasks:
    preproc.extract_info(word2concept_file='../data/flickr30k/word2concept.json', split_file='../data/flickr30k/flickr8k_test.txt')
  if 2 in tasks:
    preproc.train_test_split(split_file='../data/flickr30k/flickr8k_test.txt')
  if 3 in tasks:
    preproc.create_captions(data_file)
  if 4 in tasks:
    preproc.create_gold_alignment(data_file)
  if 5 in tasks:
    preproc.cleanup_datafile(data_file)
