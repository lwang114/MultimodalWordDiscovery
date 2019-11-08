import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
import numpy as np
from scipy.io import loadmat, wavfile
from pycocotools.coco import COCO
from PIL import Image
import random

random.seed(2)
np.random.seed(2)
#import torch
#from torchvision import transforms
#from torchvision import models

DEBUG = False
NONWORD = ['$']
NULL = 'NULL'
PUNCT = ['.', ',', '?', '!', '`', '\'', ';']
UNK = 'UNK'
DELIMITER = '('
FSAMPLE = 16000
class MSCOCO_Preprocessor():
  def __init__(self, instance_file, caption_file, pronun_dict=None):
    self.instance_file = instance_file
    self.caption_file = caption_file
    if pronun_dict is None:
      self.pronun_dict = cmudict.dict()
    else:
      self.pronun_dict = pronun_dict
    

  def extract_info(self, out_file='mscoco_info.json'):
    pair_info_list = []
    try:
      coco_api = COCO(self.instance_file)
      coco_api_caption = COCO(self.caption_file)
    except:
      raise RuntimeError("Please Run make in the cocoapi before running this") 
    
    for ex, img_id in enumerate(coco_api.imgToAnns.keys()):
      # XXX
      #if ex > 2:
      #  break
      ann_ids = coco_api.getAnnIds(img_id)
      capt_ids = coco_api_caption.getAnnIds(img_id)
      anns = coco_api.loadAnns(ann_ids)
      img_info = coco_api.loadImgs(img_id)
      img_filename = img_info[0]['file_name']
      print(img_id, img_filename)      
      
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

      # TODO: Add functionalities to extract alignment info
      pair_info = {'image_id': str(img_filename.split('.')[0]),
                   'caption_id': str(img_filename.split('.')[0]), # XXX
                   'coco_id': img_id,
                   'caption_texts': caption_list,
                   'bboxes': bboxes
                  }
      pair_info_list.append(pair_info)
  
    with open(out_file, 'w') as f:
      json.dump(pair_info_list, f, indent=4, sort_keys=True)
  
  def word_to_phones(self, sent):
    phn_seqs = []
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
    return phn_seqs
        
  def json_to_text(self, json_file, text_file, 
                  allow_repeated_concepts=False):
    json_pairs = None
    text_pairs = []
    with open(json_file, 'r') as f:
      json_pairs = json.load(f)

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
      sents = pair['caption_text'] 
      for sent in sents:
        text_pair = '%s\n%s\n' % (' '.join(concepts), sent)
        text_pairs.append(text_pair)
    
    with open(text_file, 'w') as f:
      f.write('\n'.join(text_pairs)) 
 
  # XXX: Only find the first hit
  def find_concept_occurrence_time(self, c, word_aligns):
    # TODO: verify the format of word align
    for i_w, word_align in enumerate(word_aligns):
      # Find exact match
      word = word_tokenize(word_align[0])[0] 
      if word.lower() == c:
        return word_align[0], word_align[1], word_align[2]  
       
      # Find match with related words
      if c == 'person' and word.lower() in ['people', 'man', 'men', 'woman', 'women', 'child', 'children', 'baby', 'babies', 'boy', 'girl', 'boys', 'girls']:
        return word_align[0], word_align[1], word_align[2]
      if c == 'remote' and word.lower() in ['wii', 'wiis', 'Nintendo']:
        return word_align[0], word_align[1], word_align[2]
      if c == 'airplane' and word.lower() in ['air', 'plane', 'planes']:
        if word.lower() == 'air' and i_w < len(word_aligns) - 1:
          if word_aligns[i_w+1][0] in ['plane', 'planes']:
            return word_align[0]+word_aligns[i_w+1][0], word_align[1], word_aligns[i_w+1][2] 
    return None, -1, -1
    
    return  
    # TODO: compare wordnet similarity
   
  def extract_image_audio_subset(self, json_file, max_num_per_class=200, 
                                image_base_path='./', audio_base_path='./', file_prefix='mscoco_subset'):
    with open(json_file, 'r') as f:
      pair_info = json.load(f)

    audio_dict = {}
    image_dict = {}
    phone_dict = {}
    concept2id = {}
    concept_counts = {} 
    # XXX
    for pair in pair_info:
      img_id = pair['image_id']
      bboxes = pair['bboxes']
      capt_id = pair['caption_id']
      
      print(image_base_path + str(img_id))
      img_filename = image_base_path + img_id + '.jpg'
      wav_filename = audio_base_path + capt_id + '.wav'
      img = Image.open(img_filename)    
      # XXX
      #sr, wav = wavfile.read(wav_filename)
          
      for caption in pair['caption_texts']:
        caption = word_tokenize(caption)
        # XXX
        #word_aligns = pair['word_alignment']
        word_aligns = [[word, 0, 0] for word in caption]
        concept2bbox = {}

        for bb in bboxes:
          concept = bb[0]
          c = concept.split()[-1]
          x, y, w, h = int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4])
          #print(x, y, w, h)
          word, start, end = self.find_concept_occurrence_time(c, word_aligns)

          if start != -1:
            # Extract image regions with bounding boxes
            if len(np.array(img).shape) == 2:
              region = np.tile(np.array(img)[y:y+h, x:x+w, np.newaxis], (1, 1, 3))
            else:
              region = np.array(img)[y:y+h, x:x+w, :]
            # XXX
            #segment = wav[start:end]
            
            index = sum(concept_counts.values())
            image_dict[img_id+'_'+str(index)] = region
            # XXX
            #audio_dict[capt_id+'_'+str(index)] = segment
            #print('word: ', word)
            phone_dict[capt_id+'_'+str(index)] = self.word_to_phones([word])
            if c not in concept2id:
              concept2id[c] = [[img_id+'_'+str(index), capt_id+'_'+str(index)]]
            else:
              concept2id[c].append([img_id+'_'+str(index), capt_id+'_'+str(index)])
            if c not in concept_counts:
              concept_counts[c] = 1
            elif concept_counts[c] < max_num_per_class:
              concept_counts[c] += 1

        #print(concept_counts.keys(), concept_counts.values())

    with open(file_prefix+'_concept_counts.json', 'w') as f:
      json.dump(concept_counts, f, indent=4, sort_keys=True)
    with open(file_prefix+'_concept2id.json', 'w') as f:
      json.dump(concept2id, f, indent=4, sort_keys=True)
    with open(file_prefix+'_phone.json', 'w') as f:
      json.dump(phone_dict, f, indent=4, sort_keys=True)
    # XXX
    #np.savez(file_prefix+'_audio.npz', **audio_dict)
    np.savez(file_prefix+'_image.npz', **image_dict)

  def extract_image_audio_subset_power_law(self, file_prefix='mscoco_subset', power_law_factor=1., subset_size=8000, n_concepts_per_example=5): 
    with open(file_prefix+'_concept2id.json', 'r') as f: 
      concept2ids = json.load(f)
    with open(file_prefix+'_concept_counts.json', 'r') as f: 
      concept_counts_all = json.load(f)
    with open(file_prefix+'_phone.json', 'r') as f: 
      phone_data = json.load(f)

    # XXX
    #audio_data = np.load(file_prefix+'_audio.npz')
    image_data = np.load(file_prefix+'_image.npz')
    subset_size = min(int(len(image_data.keys()) / n_concepts_per_example), subset_size)
    
    concepts = sorted(concept2ids)
    n_concept = len(concepts)
    vs = np.zeros((n_concept,))
    # TODO: Control the power law to be propto 1/n**alpha
    for i in range(n_concept):
      vs[i] = 1. / (n_concept-i) ** power_law_factor
    priors = compute_stick_break_prior(vs)
    
    image_dict = {}
    # XXX
    #audio_dict = {}
    concept_dict = {}
    phone_dict = {}
    concept_counts = {c: 0 for c in concepts}
    
    for ex in range(subset_size):
      new_data_id = 'arr_'+str(ex) 
      image_dict[new_data_id] = []
      # XXX
      #audio_dict[new_data_id] = []
      phone_dict[new_data_id] = [] 
      concept_list = []
      data_ids = []
      n_concept_remain = sum([concept_counts[c] < concept_counts_all[c] for c in concepts])
      m = min(n_concepts_per_example, n_concept_remain)
      while len(concept_list) < m:
        c = concepts[random_draw(priors)]
        if c in concept_list or concept_counts[c] >= concept_counts_all[c]:
          continue
        concept_list.append(c)
        img_id = concept2ids[c][concept_counts[c]][0]
        capt_id = concept2ids[c][concept_counts[c]][1]
        #print('img_id, image_data.keys, concept2ids[c]: ', img_id, image_data.keys(), concept2ids[c])
        image_dict[new_data_id].append(image_data[img_id])
        # XXX
        #audio_dict[new_data_id].append(audio_data[capt_id])
        phone_dict[new_data_id] += phone_data[capt_id]
        data_ids.append([img_id, capt_id])
        concept_counts[c] += 1

      concept_dict[new_data_id] = {}
      concept_dict[new_data_id]['concepts'] = concept_list  
      concept_dict[new_data_id]['data_ids'] = data_ids 
      print(concept_list)
    
    with open(file_prefix+'_concept_counts_power_law.json', 'w') as f:
      json.dump(concept_counts, f, indent=4, sort_keys=True)
    with open(file_prefix+'_concept_info_power_law.json', 'w') as f:
      json.dump(concept_dict, f, indent=4, sort_keys=True)
    with open(file_prefix+'_phone_power_law.json', 'w') as f:
      json.dump(phone_dict, f, indent=4, sort_keys=True)
    # XXX
    #np.savez(file_prefix+'_audio_power_law.npz', **audio_dict) 
    np.savez(file_prefix+'_image_power_law.npz', **image_dict)

def random_draw(p):
  x = random.random()
  n_c = len(p)
  tot = 0. 
  for c in range(n_c):
    tot += p[c]
    if tot >= x:
      return c
  return 0 

def compute_stick_break_prior(vs):
  K = len(vs)
  pvs = np.cumprod(1-vs)
  prior = np.zeros((K,))
  prior[0] = vs[0]
  prior[1:] = pvs[:-1] * vs[1:]
  return prior

if __name__ == '__main__':
  instance_file = 'annotations/instances_val2014.json'
  caption_file = 'annotations/captions_val2014.json'
  json_file = 'val_mscoco_info_text_image.json'
  image_base_path = '/home/lwang114/data/mscoco/val2014/' 
  preproc = MSCOCO_Preprocessor(instance_file, caption_file)
  #preproc.extract_info(json_file)
  #preproc.extract_image_audio_subset(json_file, image_base_path=image_base_path)
  preproc.extract_image_audio_subset_power_law()
