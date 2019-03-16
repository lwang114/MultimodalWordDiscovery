import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
import numpy as np
from scipy.io import loadmat
#from pycocotools.coco import COCO
from PIL import Image
#import torch
#from torchvision import transforms
#from torchvision import models

DEBUG = True
NULL = 'NULL'
UNK = 'UNK'
PUNCT = ['.', ',', '?', '!', '`', '\'', ';'] #'\"']

# TODO: Add config options
'''class MSCOCO_Preprocessor():
  def __init__(self, instance_file, caption_file, pronun_dict=None):
    self.instance_file = instance_file
    self.caption_file = caption_file
    self.pronun_dict = pronun_dict

  def extract_info(self, out_file='mscoco_info.json'):
    pair_info_list = []
    try:
      coco_api = COCO(self.instance_file)
      coco_api_caption = COCO(self.caption_file)
    except:
      raise RuntimeError("Please Run make in the cocoapi before running this") 
    
    for img_id in coco_api.imgToAnns.keys():
      print(img_id)      
      ann_ids = coco_api.getAnnIds(img_id)
      capt_ids = coco_api_caption.getAnnIds(img_id)
      anns = coco_api.loadAnns(ann_ids)
      captions = coco_api_caption.loadAnns(capt_ids)
      bboxes = []
      
      for ann in anns:
        cat = coco_api.loadCats(ann['category_id'])[0]['name']
        if DEBUG:
          print(ann)
        x, y, w, h = ann['bbox']
        # If the concept class is a compound, combine the words
        if len(cat.split()) > 1:
          cat = '_'.join(cat.split())
        
        bboxes.append((cat, x, y, w, h))
      
      caption_list = []
      for capt in captions:
        if DEBUG:
          print(capt)
        caption = capt['caption']
        caption = ' '.join(word_tokenize(caption))
        caption_list.append(caption)

      pair_info = {'img_id': img_id,
                  'text': caption_list,
                  'bboxes': bboxes
                  }
      pair_info_list.append(pair_info)
  
    with open(out_file, 'w') as f:
      json.dump(pair_info_list, f)

  def word_to_phoneme(self, sent):
    pass

  def json_to_text(self, json_file, text_file, 
                  allow_repeated_concepts=False):
    json_pairs = None
    text_pairs = []
    with open(json_file, 'r') as f:
      json_pairs = json.load(f)
    
    ### Temporary: comment this line out once the json file is in proper format
    #json_pairs = json_pairs['data']

    for pair in json_pairs:
      concepts = []
      sent = '' 
      bboxes = pair['bboxes']
      for bb in bboxes:
        concept = bb[0]
        concepts.append(concept)
      
      if not allow_repeated_concepts:
        concepts = list(set(concepts))

      # TODO: Retokenize
      sents = pair['text'] 
      for sent in sents:
        text_pair = '%s\n%s\n' % (' '.join(concepts), sent)
        text_pairs.append(text_pair)
    
    with open(text_file, 'w') as f:
      f.write('\n'.join(text_pairs))   
'''

class Flickr_Preprocessor(object):
  def __init__(self, instance_file, alignment_file, caption_file, image_path='./', category_file='imagenet_class_index.json'):
    self.img_path = image_path
    self.bbox_info = loadmat(instance_file)['bboxes_arr']
    self.align_file = alignment_file
    with open(caption_file, 'r') as f:
      self.captions_list = f.read().strip().split('\n')

    with open(alignment_file, 'r') as f:
      self.align_list = f.read().strip().split('\n')
    
    with open(category_file, 'r') as f:
      class_idx = json.load(f)
      self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))] 
    
    self.pronun_dict = cmudict.dict()

  def extract_info(self, out_file='flickr.txt'):
    pairs = []
    for entry in self.captions_list:
      parts = entry.split()
      img_capt_id = parts[0]
      sent = ' '.join(parts[1:])  
      parts = img_capt_id.split('#')
      img_filename, capt_id = parts[0], parts[1]
      img_id = img_filename.split('.')[0]
      if DEBUG:
        print(img_id)

      if capt_id == '0':
        aligns = self._getAlignmentFromId(img_id)
        ### Temporary: For datasets with annotations on every image, comment this out
        if not aligns:
          continue
        if DEBUG:
          print(aligns)
        img_concepts = []
        for align in aligns:
          ### To use VGG 16 to find the image label, uncomment this line
          #img_concept = self._getImgConcept(align[1:], img_filename) 
          img_concept = self._getImgConceptWordnet(align)
          if not img_concept:
            continue
          img_concepts.append([img_concept] + align)

        pair = {'image_id': img_id,
                'image_filename': img_filename,
                'caption_texts': [sent],
                'image_concepts': img_concepts,
        }
        pairs.append(pair)
      else:
        for i, pair in enumerate(pairs):
          if pair['image_filename'] == img_id:
            pairs[i]['caption_texts'].append(sent)

    with open(out_file, 'w') as f:
      json.dump(pairs, f, indent=4, sort_keys=True)

  def train_test_split(self, in_file, n_test_example, indices=None):
    with open(in_file, 'r') as f:
      data = f.read().split('\n')

    test_data = ['%s\n%s\n' % (data[3*i], data[3*i+1]) if not indices else indices[i] for i in range(n_test_example)]
    train_data = data[3*n_test_example:]
    with open(in_file+'.train', 'w') as f:
      f.write('\n'.join(train_data))
    with open(in_file+'.test', 'w') as f:
      f.write('\n'.join(test_data))

  def train_test_split_from_file(self, in_file, test_file_list, out_file='flickr30k', indices=None):
    with open(in_file, 'r') as f:
      data = json.load(f)

    with open(test_file_list, 'r') as f:
      test_files = f.read().split('\n')

    train_data = []
    test_data = []
    test_ids = [f.split('_')[0] for f in test_files]
    for pair in data:
      concepts = [bbox_info[0] for bbox_info in pair['image_concepts']]
      concepts = ' '.join(sorted(list(set(concepts))))
      if pair['image_id'] in test_ids:
        for caption in pair['caption_phonemes']:
          test_data.append('%s\n%s\n' % (concepts, caption))
      else:
        for caption in pair['caption_phonemes']:
          train_data.append('%s\n%s\n' % (concepts, caption))
    print('Number of test data: ', len(test_data))
    with open(out_file+'.train', 'w') as f:
      f.write('\n'.join(train_data))
    with open(out_file+'.test', 'w') as f:
      f.write('\n'.join(test_data))


  # XXX: Only works for English
  def word_to_phoneme(self, in_file, out_file='phoneme_concept_info.json'):
    with open(in_file, 'r') as f:
      data_info = json.load(f)
    
    for i in range(len(data_info)): 
      sents = data_info[i]['caption_texts']
      data_info[i]['caption_phonemes'] = []
      for sent in sents:
        phn_seqs = []
        # XXX: For flickr, this is fine
        sent = sent.split() #word_tokenize(sent)
        for word in sent: 
          if word in PUNCT:
            continue
          if word.lower() in self.pronun_dict:
            phns = self.pronun_dict[word.lower()][0]
          else:
            if DEBUG:
              print(word)
            phns = [UNK] 
          phn_seqs += phns
        
        data_info[i]['index'] = i
        data_info[i]['caption_phonemes'].append(' '.join(phn_seqs))

    with open(out_file, 'w') as f:
      json.dump(data_info, f, indent=4, sort_keys=True) 

  def json_to_text(self, json_file, text_file, 
                  allow_repeated_concepts=False):
    json_pairs = None
    text_pairs = []
    with open(json_file, 'r') as f:
      json_pairs = json.load(f)
    
    # Temporary: comment this line out once the json file is in proper format
    #json_pairs = json_pairs['data']

    for pair in json_pairs:
      concepts = []
      sent = '' 
      bboxes = None
      if 'bboxes' in pair.keys():
        bboxes = pair['bboxes']
      else:
        bboxes = pair['image_concepts']

      for bb in bboxes:
        concept = bb[0]
        concepts.append(concept)
      
      if not allow_repeated_concepts:
        concepts = sorted(list(set(concepts)))
        if DEBUG:
          print('concept for text file: ', concepts)

      # TODO: Retokenize
      sents = None
      if 'text' in pair.keys():
        sents = pair['text'] 
      elif 'caption_phonemes' in pair.keys():
        sents = pair['caption_phonemes']
      else:
        sents = pair['caption_texts']

      for sent in sents:
        text_pair = '%s\n%s\n' % (' '.join(concepts), sent)
        text_pairs.append(text_pair)
    
    with open(text_file, 'w') as f:
      f.write('\n'.join(text_pairs))   

  def json_to_xnmt_text(self, json_file, text_file, 
                  allow_repeated_concepts=False):
    json_pairs = None
    text_pairs = []
    with open(json_file, 'r') as f:
      json_pairs = json.load(f)
    
    # Temporary: comment this line out once the json file is in proper format
    #json_pairs = json_pairs['data']
    src = []
    trg = []
 
    for pair in json_pairs:
      concepts = []
      sent = '' 
      bboxes = None
      if 'bboxes' in pair.keys():
        bboxes = pair['bboxes']
      else:
        bboxes = pair['image_concepts']

      for bb in bboxes:
        concept = bb[0]
        concepts.append(concept)
      
      if not allow_repeated_concepts:
        concepts = sorted(list(set(concepts)))
      
      # TODO: Retokenize
      sents = None
      if 'text' in pair.keys():
        sents = pair['text'] 
      elif 'caption_phonemes' in pair.keys():
        sents = pair['caption_phonemes']
      else:
        sents = pair['caption_texts']

      for sent in sents:
        single_src = sent
        single_trg = ' '.join(concepts) #NULL + ' ' + ' '.join(concepts)
        
        src.append(single_src)
        trg.append(single_trg)
    
    with open('src_' + text_file, 'w') as f:
      f.write('\n'.join(src))
    with open('trg_' + text_file, 'w') as f:
      f.write('\n'.join(trg))

  # Does not allow repeated concept for now
  def create_gold_alignment(self, data_file, out_file='gold_align.json', is_phoneme=True):
    with open(data_file, 'r') as f:
      data = json.load(f)
    
    aligns = []
    for i, datum in enumerate(data):
      # Temporary: Change it to phoneme sequences later
      sents = None
      if 'caption_phonemes' in datum:
        sents = datum['caption_phonemes'] 
      else:
        sents = datum['caption_texts'] 

      img_concepts = datum['image_concepts']
      
      sents = [s.split() for s in sents]
      for sent in sents:
        align = [0]*len(sent)
        concepts = []
        for c in img_concepts:
          concepts.append(c[0])
        concepts = sorted(list(set(concepts)))
        concepts = [NULL] + concepts
        #if DEBUG:
        #  print('concepts for alignment', concepts)
      
        concept2pos = {c:i for i, c in enumerate(concepts)}

        for c in img_concepts:
          phrase = c[1].split()
          if is_phoneme:
            phrase = self._toPhoneme(phrase)
          start = self._findTokenIdx(sent, phrase) 
          if DEBUG:
            if i == 1745:
              print('image concept, start idx: ', sent, phrase, start)
          if start == -1:
            continue
          
          for j in range(len(phrase)):
            #if DEBUG:
            #  print('len(sent): ', len(sent))
            #  print('align indices: ', start, start+j)
              
            align[start+j] = concept2pos[c[0]] 
        align_info = {'index': i,
                      'caption': sent,
                      'image_concepts': concepts, 
                      'alignment': align,
                      'is_phoneme': is_phoneme
                      }
        aligns.append(align_info) 

    if out_file.split('.')[-1] == 'json':
      with open(out_file, 'w') as f:
        json.dump(aligns, f, indent=4, sort_keys=True)
    # Create .ref file for xnmt
    elif out_file.split('.')[-1] == 'ref':
      with open(out_file, 'w') as f:
        for align_info in aligns:
          ali = align_info['alignment']
          trg_words = align_info['image_concepts']
          alignment = ' '.join([trg_words[idx] for idx in ali])
          f.write("%s\n" % alignment)

  def alignment_to_clusters(self, alignment_file, 
                                  cluster_file='clusters.json', 
                                  phoneme_level=False):
    def _find_distinct_tokens(data):
      tokens = set()
      for datum in data:
        if 'image_concepts' in datum: 
          tokens = tokens.union(set(datum['image_concepts']))
        elif 'foreign_sent' in datum:
          tokens = tokens.union(set(datum['foreign_sent']))

      return list(tokens)
    
    with open(alignment_file, 'r') as f:
      aligns = json.load(f)
    
    image_concepts = _find_distinct_tokens(aligns)
    clusters = {c:[] for c in image_concepts}
    
    for align_info in aligns:
      align = align_info['alignment']
      sent = None
      if 'caption_text' in align_info:
        sent = align_info['caption_text']
        concepts = None
        if 'image_concepts' in align_info:
          concepts = align_info['image_concepts']
        #sent = word_tokenize(sent)
        for i, w in enumerate(sent):
          c = concepts[align[i]]
          clusters[c].append(w)
          clusters[c] = list(set(clusters[c]))
      
      # Temporary: Change names later
      elif 'target_sent' in align_info:
        sent = align_info['target_sent']
        concepts = None
        if 'foreign_sent' in align_info:
          concepts = align_info['foreign_sent']
        else:
          raise TypeError('Wrong file format')
     
        for i, w in enumerate(sent):
          c = concepts[align[i]]
          clusters[c].append(w)
          clusters[c] = list(set(clusters[c])) 
 
      # TODO: For phoneme sequences
      elif 'caption' in align_info:
        sent = align_info['caption']
        concepts = None
        if 'image_concepts' in align_info:
          concepts = align_info['image_concepts']
        #sent = word_tokenize(sent)
        for i, w in enumerate(sent):
          c = concepts[align[i]]
          clusters[c].append(w)
          clusters[c] = list(set(clusters[c]))
      
    with open(cluster_file, 'w') as f:
      json.dump(clusters, f, indent=4, sort_keys=True)

  def _toPhoneme(self, sent):
    phn_seqs = []
    for word in sent: 
      if word in PUNCT:
        continue
      if word.lower() in self.pronun_dict:
        phns = self.pronun_dict[word.lower()][0]
      else:
        #if DEBUG:
        #  print(word)
        phns = [UNK] 
      phn_seqs += phns
    return phn_seqs

  # Find the position of a token in a sentence   
  def _findTokenIdx(self, sent, token):
      sent = ' '.join(sent)
      token = ' '.join(token)
      start_str_idx = sent.find(token)
      if start_str_idx == -1:
        return -1
      # May need a tokenizer for other dataset
      sent_tokenize = sent.split() #word_tokenize(sent)
      # WARNING: Only works if the delimiter is space
      l = 0
      for i, w in enumerate(sent_tokenize):
        l += len(w) + 1
        if l > start_str_idx:
          return i 
          
  def _getAlignmentFromId(self, img_id):
    aligns = []
    for i, entry in enumerate(self.align_list):
      cur_id = entry.split()[0].split('_')[0]
      #if DEBUG:
      #  if i % 10 == 0:
      #    print(img_id, cur_id) 
      if cur_id == img_id:
        align = ' '.join(entry.split()[1:])
        bbox = self.bbox_info[i]
        # Only works for bounding box on .jpg file
        aligns.append([align, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    
    return aligns

  # Get the category of a bounding box using VGG16 classifier 
  def _getImgConcept(self, bbox, img_file):
    # TODO: Double check this
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    img = Image.open('%s%s' % (self.img_path, img_file))
    img = np.array(img)

    # TODO: Double check this
    region = img[y:y+h, x:x+w, :]
    region = Image.fromarray(region)
  
    resize_crop = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor()]       
    )
    
    img = resize_crop(region)
    Vgg16 = models.vgg16(pretrained=True)
    out = Vgg16(img.unsqueeze(0))
    idx = out[0].sort()[1][-1]
    return self.idx2label[idx]
    #return bbox

  # Get the category of a bounding box using wordnet  
  def _getImgConceptWordnet(self, align):
    noun_phrase = align[0]
    noun_phrase = word_tokenize(noun_phrase)
    pos = nltk.pos_tag(noun_phrase)[::-1]
    
    ### Find the first noun in the phrase starting 
    ### from the end of the phrase
    nouns = []
    concept = None
    for w in pos:
      if w[1][0] == 'N':
        nouns.append(w[0])
    
    if len(nouns) == 0:
      print('Concept not found')
      return
    else:
      i = 0
      found = 0
      while not found and i < len(nouns):
        concept = wn.synsets(nouns[i])
        if concept:
          concept = concept[0].name()
          found = 1
        i += 1
      if not found:
        print('Concept not found')
        return

    if DEBUG:
      print(concept)
    return concept.split('.')[0]
    
if __name__ == '__main__':
  '''instance_file = 'annotations/instances_val2014.json'
  caption_file = 'annotations/captions_val2014.json'
  json_file = 'val_mscoco_info_text_image.json'
  preproc = MSCOCO_Preprocessor(instance_file, caption_file)
  preproc.extract_info(json_file)
  preproc.json_to_text(json_file, 'mscoco_val.txt')
  '''
  datapath = '../data/' 
  instance_file = datapath + 'flickr30k/bboxes.mat'
  alignment_file = datapath + 'flickr30k/flickr30k_phrases.txt'
  caption_file = datapath + 'flickr30k/results_20130124.token'
  json_file = datapath + 'flickr30k/word_level/flickr30k_info_text_concept.json'
  category_file = datapath + 'imagenet_class_index.json'
  preproc = Flickr_Preprocessor(instance_file, alignment_file, caption_file, category_file=category_file, image_path='../../data/flickr30k/flickr30k-images/')
  #preproc.train_test_split(datapath+'flickr30k/phoneme_level/flickr30k.txt', 100)
  preproc.train_test_split_from_file(datapath+'flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json', test_file_list='/Users/liming/research/data/flickr/flickr8k_test.txt')

  '''
  preproc.extract_info(json_file)
  preproc.json_to_text(json_file, 'flickr30k.txt')
  preproc.json_to_xnmt_text(json_file, 'flickr30k.txt')

  preproc.create_gold_alignment(json_file, 'flickr30k_gold_alignment.json')
  preproc.create_gold_alignment(json_file, 'flickr30k_alignment.ref')
  preproc.word_to_phoneme(json_file, 'flickr30k_info_phoneme_concept.json')
  preproc.json_to_text(datapath + 'flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json', 'flickr30k.txt')
  preproc.json_to_xnmt_text('flickr30k_info_phoneme_concept.json', 'flickr30k.txt')
  preproc.create_gold_alignment(datapath + 'flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json', datapath + 'flickr30k/phoneme_level/flickr30k_gold_alignment.json') 
  preproc.create_gold_alignment(datapath + 'flickr30k/word_level/flickr30k_info_text_concept.json', datapath + 'flickr30k/word_level/flickr30k_gold_alignment.json')
  preproc.alignment_to_clusters(datapath + 'flickr30k/word_level/flickr30k_gold_alignment.json', datapath + 'flickr30k/word_level/flickr30k_gold_clusters.json')
  preproc.alignment_to_clusters(datapath + 'flickr30k/phoneme_level/flickr30k_gold_alignment.json', datapath + 'data/flickr30k/phoneme_level/flickr30k_gold_clusters.json')'''
  #preproc.alignment_to_clusters('../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_alignment.json', '../smt/exp/ibm1_phoneme_level_clustering/flickr30k_pred_clusters.json')
   
