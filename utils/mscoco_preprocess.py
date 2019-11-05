import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
import numpy as np
from scipy.io import loadmat, wavfile
from pycocotools.coco import COCO
from PIL import Image
#import torch
#from torchvision import transforms
#from torchvision import models

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
    with open(in_file, 'r') as f:
      data_info = json.load(f)
    
    for i in range(len(data_info)): 
      sents = data_info[i]['caption_texts']
      data_info[i]['caption_phonemes'] = []
      for sent in sents:
        phn_seqs = []
        sent = word_tokenize(sent)
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
 
  # XXX: Only find the first hit
  def find_concept_occurrence_time(self, c, word_aligns):
    # TODO: verify the format of word align
    for word_align in word_aligns:
      # Find exact match
      word = word_tokenize(word_aligns[0])
      if word.lower() == c:
        return word_align 
       
      # Find match with related words
      if c == 'remote' and word.lower() in ['wii', 'wiis', 'Nintendo']:
        return word_align
      if c == 'airplane' and word.lower() in ['air', 'plane', 'planes']:
        return word_align 
    
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
    for pair in pair_info[:2]:
      concepts = []
      img_id = pair['img_id']
      bboxes = pair['bboxes']
      capt_id = pair['capt_id']
      
      img_filename = image_base_path + img_id + '.jpg'
      wav_filename = audio_base_path + capt_id + '.wav'
      img = Image.open(img_filename)    
      sr, wav = wavfiles.read(wav_filename)
          
      caption = word_tokenize(pair['captions'])
      # XXX
      #word_aligns = pair['word_alignment']
      word_aligns = [[word, 0, 0] for word in caption]
      concept2bbox = {}

      for bb in bboxes:
        concept = bb[0]
        
        # Use the last word as concept label if it is a compound
        if len(concept.split()) > 1:
          concept = concept[-1]
        concepts.append(concept)
        if concept not in concept2bbox:
          concept2bbox[concept] = [bb]
        else:
          concept2bbox[concept].append(bb) 
      
      print('number of concepts: ', len(concept2bbox.keys()))
      
      for c, bb in bboxes.items():
        x, y, w, h = bb[1], bb[2], bb[3], bb[4]
        # XXX: Uncomment for audio
        word, start, end = self.find_concept_occurrence_time(c, word_aligns)

        if start != -1:
          # Extract image regions with bounding boxes
          region = np.array(img[y:y+h, x:x+w, :])
          # XXX
          #segment = wav[start:end]
          
          index = sum(concept_counts.values())
          image_dict[img_id+'_'+str(index)] = region
          # XXX
          #audio_dict[capt_id+'_'+str(index)] = segment 
          phone_dict[capt_id+'_'+str(index)] = self.word_to_phone(word) 
          if c not in concept2id:
            concept2id[c] = [capt_id+'_'+str(index)]
          else:
            concept2id[c].append(capt_id+'_'+str(index))
          if c not in concept_counts:
            concept_counts[c] = 1
          elif concept_counts[c] < max_num_per_class:
            concept_counts[c] += 1

    with open(file_prefix+'_concept_counts.json', 'w') as f:
      json.dump(concept_counts, f)
    with open(file_prefix+'_concept2id.json', 'w') as f:
      json.dump(text_dict, f)
    with open(file_prefix+'_phone.json', 'w') as f:
      json.dump(phone_dict, f)
    # XXX
    #np.savez(file_prefix+'_audio.npz', 'w')
    np.savez(file_prefix+'_image.npz', 'w')

  def extract_image_audio_subset_power_law(self, file_prefix, power_law_factor=1., subset_size=4000, n_concepts_per_example): 
    with open(file_prefix+'_concept2id.json', 'r') as f: 
      concept2ids = json.load(f)
    with open(file_prefix+'_concept_counts.json', 'r') as f: 
      concept_counts_all = json.load(f)
    with open(file_prefix+'_phone.json', 'r') as f: 
      phone_data = json.load(f)

    audio_data = np.load(file_prefix+'_audio.npz')
    image_data = np.load(file_prefix+'_image.npz')

    concepts = sorted(concept2ids)
    n_concept = len(concepts)
    vs = np.zeros((n_concept,))
    # TODO: Control the power law to be propto 1/n**alpha
    for i in range(n_concept):
      vs[i] = 1. / (n-i) ** power_law_factor
    priors = compute_stick_breaking_prior(vs)
    
    image_dict = {}
    audio_dict = {}
    concept_dict = {}
    phone_dict = {}
    concept_counts = {c: 0 for c in concepts}
    for ex in range(subset_size):
      new_data_id = '_'.join('arr_'+[str(ex)]) 
      image_dict[new_data_id] = []
      audio_dict[new_data_id] = []
      phone_dict[new_data_id] = '' 
      concept_list = []
      data_ids = []
      for len(concept_list) < n_concepts_per_example:
        c = random_draw(priors)
        if c in concept_list or concept_counts[c] >= concept_counts_all[c]:
          continue
        concept_list.append(c)
        data_id = concept2ids[c][concept_counts[c]]
        image_dict[new_data_id].append(image_data[data_id])
        audio_dict[new_data_id].append(audio_data[data_id])
        phone_dict[new_data_id] += phone_data[data_id]+' '
        data_ids.append(data_id)
        concept_counts[c] += 1

      concept_dict[new_data_id] = {}
      concept_dict[new_data_id]['concepts'] = concept_list  
      concept_dict[new_data_id]['data_ids'] = data_ids 
      print(concept_list)
    
    with open(file_prefix+'_concept_counts_power_law.json', 'w') as f:
      json.dump(concept_counts, f)
    with open(file_prefix+'_concept_info_power_law.json', 'w') as f:
      json.dump(concept_dict, f)
    with open(file_prefix+'_phone_power_law.json', 'w') as f:
      json.dump(phone_dict, f)
    np.savez(file_prefix+'_audio_power_law.npz', 'w')
    np.savez(file_prefix+'_image_power_law.npz', 'w')

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
