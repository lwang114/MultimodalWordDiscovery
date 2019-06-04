import numpy as np
import json
import os
import time
from sklearn.cluster import MiniBatchKMeans
from nltk.corpus import cmudict

DEBUG = False
NONWORD = ['$']
PUNCT = ['.', ',', '?', '!', '`', '\'', ';']
UNK = 'UNK'
DELIMITER = '('
   
'''class WordSegmentationLoader:
  def __init__(self, word_align_dir, word_align_info_file=None):
    self.word_align_dir = word_align_dir
    self.word_align_files = os.listdir(word_align_dir)
    self.word_align_info = {}
    if word_align_info_file:
      with open(word_align_info_file, 'r') as f:
        self.word_align_info = json.load(f) 

  # Load the phone segmentation into a .txt or .json file
  # TODO: Make this more general (now works only for articulatory feature alignment)
  def extract_info(self, out_file='word_alignment.json'):
    begin = time.time()
    for f in self.word_align_files:
      img_id, _, capt_id = f.split('.')[0].split('_')
      if img_id not in self.word_align_info:
        self.word_align_info[img_id] = {}
      self.word_align_info[img_id][capt_id] = []
      
      fp = open(word_align_dir + f, 'r')
      for line in fp:
        word, st, end = line.strip().split(' ') 
        if len(set(NONWORD).intersection(set(word))) > 0:
          continue
        if DELIMITER in word:
          word = word.split(DELIMITER)[0]
        
        self.word_align_info[img_id][capt_id].append((word, float(st), float(end))) 
      fp.close()  

      with open(out_file, 'w') as f:
        json.dump(self.word_align_info, f, indent=4, sort_keys=True)
      print('Takes %0.5f to extract the word segmentations into json: ', time.time() - begin)

  def extract_info_from_id(self, img_id):
    word_align_info = {} 

    for f in self.word_align_files:
      cur_img_id, _, capt_id = f.split('.')[0].split('_')
      if cur_img_id != img_id:
        continue

      fp = open(self.word_align_dir + f, 'r')
      caption_info = []
      for line in fp:
        word, st, end = line.strip().split(' ') 

        if len(set(NONWORD).intersection(set(word))) > 0:
          continue
        if DELIMITER in word:
          word = word.split(DELIMITER)[0]
        caption_info.append((word, float(st), float(end)))
      fp.close()
      word_align_info[capt_id] = caption_info 
        
    return word_align_info
      
  # Create an audio-to-concept alignment from word-to-concept alignment 
  def generate_gold_audio_concept_alignment(self, word_concept_align_file, feat_info_file, out_file='gold_alignment.json'):
  '''

class FlickrBottleneckPreprocessor:
  def __init__(self, bnf_train_file, bnf_test_file, info_file, word_align_dir, phone_centroids_file=None, caption_seqs_file=None, n_clusters=60, batch_size=100):
    self.bnf_train = np.load(bnf_train_file)
    self.bnf_test = np.load(bnf_test_file)
    self.pronun_dict = cmudict.dict() 
    self.word_align_dir = word_align_dir
    # Load the .npz file
    self.cluster_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    self.n_clusters = n_clusters
    # Cluster features
    if phone_centroids_file:
      cluster_centers = np.load(phone_centroids_file)
      self.cluster_model.cluster_centers_ = cluster_centers
    

    with open(info_file, 'r') as f:
      self.data_info = json.load(f)
    
    self.captions = {}
    if caption_seqs_file:
      with open(caption_seqs_file, 'r') as f:
        self.captions = json.load(f)
      
  # TODO: Avoid loading the whole data to disk
  def generate_cluster(self, out_file='pseduophns'):
    train_frames = []
    for k in self.bnf_train.keys():
      frames = np.split(self.bnf_train[k], self.bnf_train[k].shape[0], axis=0)
      train_frames += frames
    if DEBUG:
      print('train_frames: ', np.array(train_frames).squeeze().shape)
    self.cluster_model.fit(np.array(train_frames).squeeze())
    np.save(out_file, self.cluster_model.cluster_centers_)

  def label_caption(self, caption):
    cluster_ids = self.cluster_model.predict(caption)
    #merged_cluster_ids = self._merge_repeated_frames(cluster_ids.tolist())
    #return merged_cluster_ids
    return cluster_ids.tolist()

  def label_captions(self, out_file='caption_seqs.json'):
    for feat_id in sorted(self.bnf_train.keys()): 
      if feat_id not in self.captions:
        self.captions[feat_id] = {}
      self.captions[feat_id]['caption_sequences'] = self.label_caption(self.bnf_train[feat_id])
      if DEBUG:
        print('bnf_train[feat_id].shape: ', self.bnf_train[feat_id].shape, len(self.captions[feat_id]['caption_sequences']))
      self.captions[feat_id]['is_train'] = True

    for feat_id in sorted(self.bnf_test.keys()): 
      if feat_id not in self.captions:
        self.captions[feat_id] = {}
      
      self.captions[feat_id]['caption_sequences'] = self.label_caption(self.bnf_test[feat_id])
      self.captions[feat_id]['is_train'] = False

    with open(out_file, 'w') as f:
      json.dump(self.captions, f, indent=4, sort_keys=True)
  
  def _concat_word_bn_seq(self, bn_seq, word_seq):
    nframes = len(bn_seq)
    tot_time = word_seq[-1][2]
    concat_seq = ''
    for w in word_seq:
      st_frame, end_frame = int(float(nframes) * w[1] / tot_time), int(float(nframes) * w[2] / tot_time)
      for i in range(st_frame, end_frame):
        concat_seq += '_'.join([str(bn_seq[i]), w[0]]) + ' '
    return concat_seq

  def extract_info(self, out_file='bnf_info.json'):
    bnf_data_info = {}
    # TODO: Split the data into training and testing
    if not self.cluster_model.cluster_centers_.all(): 
      self.generate_cluster()

    if not self.captions:
      self.label_captions()
    
    # Loop through all the files in the json info dict
    for pair in enumerate(sorted(self.data_info, key=lambda x:x['index'])):
      print('Index: ', pair['index'])      
      img_filename = pair['image_filename'] 
      img_id = pair['image_id']
      # Create a new entry for the new json info file
      # Find the image id, image filename, caption text, caption phonemes, image_concepts from the json info
      # TODO: Make the file more space-efficient 
      if img_id not in bnf_data_info:
        bnf_data_info[img_id] = {}
        bnf_data_info[img_id]['image_filename'] = img_filename
        bnf_data_info[img_id]['index'] = pair['index']
        bnf_data_info[img_id]['image_concepts'] = pair['image_concepts']
        bnf_data_info[img_id]['caption_phonemes'] = {}  
        bnf_data_info[img_id]['caption_texts'] = {}  
        bnf_data_info[img_id]['caption_sequences'] = {} 

      for feat_id, caption_info in self.captions.items():
        cur_img_id, _, capt_id, _ = feat_id.split('_') 
        if cur_img_id == img_id: 
          bnf_data_info[img_id]['caption_texts'][capt_id] = []
          bnf_data_info[img_id]['caption_phonemes'][capt_id] = []
          bnf_data_info[img_id]['is_train'] = caption_info['is_train']    
          ali_f = '_'.join(feat_id.split('_')[:3]) + '.words'
          if os.path.isfile(self.word_align_dir + ali_f):
            print(self.word_align_dir + ali_f)
            # Extract captions in words and phone sequence
            fp = open(self.word_align_dir + ali_f, 'r')
     
            for line in fp:
              word, st, end = line.strip().split(' ') 
              if DELIMITER in word:
                word = word.split(DELIMITER)[0]    
              if len(set(NONWORD).intersection(set(word))) > 0:
                continue
              
              bnf_data_info[img_id]['caption_texts'][capt_id].append((word, float(st), float(end)))
            
              if word.lower() in self.pronun_dict: 
                bnf_data_info[img_id]['caption_phonemes'][capt_id] += self.pronun_dict[word.lower()][0] 
              else:
                bnf_data_info[img_id]['caption_phonemes'][capt_id] += [UNK] 
            
            if len(bnf_data_info[img_id]['caption_texts'][capt_id]) == 0:
              print('word segmentation file does not exist')
              continue

            #if DEBUG:
            #  print('caption texts: ', bnf_data_info[img_id]['caption_texts'].values())
            #  print('caption phonemes: ', bnf_data_info[img_id]['caption_phonemes'].values()) 
            fp.close()  
      
            # Extract captions in cluster sequence concatenated with word sequence for readability
            bnf_data_info[img_id]['caption_sequences'][capt_id] = self._concat_word_bn_seq(caption_info['caption_sequences'], 
                                                                                     bnf_data_info[img_id]['caption_texts'][capt_id]) 
   
    with open(out_file, 'w') as fp:
      # Save the json info file
      json.dump(bnf_data_info, fp, indent=4, sort_keys=True)
  
  # TODO: create gold alignment from info file
  #def create_gold_alignment(self):

  def create_gold_alignment(self, data_info_file, word_concept_align_file, out_file='gold_alignment.json'):
    begin = time.time()
    audio_concept_alignments = []
    with open(data_info_file, 'r') as f:
      data_info = json.load(f)

    with open(word_concept_align_file, 'r') as f:
      word_concept_align_info = json.load(f)

    # Loop through each caption-concept pair in the word-level info database
    for info in sorted(data_info.values(), key=lambda x: x['index']):
      idx = info['index']
      print('Processing index:', idx)
      img_id = info['image_filename'].split('.')[0]
      word_concept_align = word_concept_align_info[idx]['alignment']
      img_concepts = word_concept_align_info[idx]['image_concepts']
      concept_order = {c:i for i, c in enumerate(sorted(img_concepts))}
      sent = [w.lower() for w in word_concept_align_info[idx]['caption_text'][:-1] if len(w) > 1 or (w[0].lower() >= 'a' and w[0].lower() <= 'z')]
      #if img_id not in word_concept_align_info:
      #  print('Image id %s not found' % img_id)
      #  continue
       
      # Find the corresponding caption in the word alignment dictionary by matching the words with the image concepts
      # TODO: deal with inexact match (flickr30k has only one caption per image while flickr8k has five per image)
      word_aligns = info['caption_texts']
      concepts = [c[0] for c in info['image_concepts']]

      word_align = None
      best_IoU = 0.
      best_capt_id = None

      for capt_id, cur_word_align in word_aligns.items(): 
        cur_sent = [a[0].lower() for a in cur_word_align]
        
        #if DEBUG:
        #  print('sent: ', sent)
        #  print('cur_sent: ', cur_sent)
        #  print('sent intersect cur_sent', set(cur_sent).intersection(set(sent)))
        cur_IoU = float(len(set(cur_sent).intersection(set(sent)))) / float(len(set(cur_sent).union(set(sent))))
        if cur_IoU > best_IoU:
          best_IoU = cur_IoU
          best_capt_id = capt_id
      
      if not best_capt_id:
        print('caption not found')
        continue
      
      word_align = word_aligns[best_capt_id] 
      cur_sent = [a[0].lower() for a in word_align]
      trg2ref, ref2trg = self._map_transcripts(cur_sent, sent)
      if DEBUG:
        print('trg: ', cur_sent)
        print('ref: ', sent)
        print('trg2ref', trg2ref)
        print('ref2trg', ref2trg)
        
      # Convert the time stamps to frame numbers
      nframes = len(info['caption_sequences'][best_capt_id].split()) 

      T = word_align[-1][-1] 
      audio_concept_align = [0]*nframes 
      
      for pos, a in enumerate(word_align):
        word, st, end = a[0], a[1], a[2]
        st_frame, end_frame = int(st * float(nframes) / T), int(end * float(nframes) / T) 
        end_frame = min(end_frame, nframes)
        for i_f in range(st_frame, end_frame):
          if len(trg2ref[pos]) > 1:
            for r_pos in trg2ref[pos]:
              audio_concept_align[i_f] = word_concept_align[r_pos]
          # Remove this case if compound words only appear in isolation  
          elif len(trg2ref[pos]) == 0:
            continue
          else:
            cur_concept = img_concepts[word_concept_align[trg2ref[pos][0]]]
            audio_concept_align[i_f] = concept_order[cur_concept]
      if DEBUG:
        print('nframes: ', nframes)
        print('concept indices: ', set(audio_concept_align))
       
      # Create an entry for the current caption-concept pair
      audio_concept_info = {
          'alignment': audio_concept_align,
          'caption_text': sent,
          'image_concepts': sorted(img_concepts),
          'image_id': img_id,
          'capt_id': best_capt_id
        }
      audio_concept_alignments.append(audio_concept_info)
      with open(out_file, 'w') as f:
        json.dump(audio_concept_alignments, f, indent=4, sort_keys=True)
    print('Take %0.5f to generate audio-concept alignment', time.time() - begin)

  # TODO: Multiple captions for one image (so do not need to use the gold alignment file)
  def json_to_xnmt_format(self, bnf_data_info_file, gold_align_file, out_prefix='flickr_bnf', train_test_split=False):
    bnf_data_info = None
    with open(bnf_data_info_file, 'r') as f:
      bnf_data_info = json.load(f)
    
    if train_test_split:
      train_trg = open(out_prefix+'_train_trg.txt', 'w')
      test_trg = open(out_prefix+'_test_trg.txt', 'w')
    
      # Generate the image concept (.txt) file
      for img_id, pair in sorted(bnf_data_info.items(), key=lambda x:x[1]['index']):
        img_concepts = pair['image_concepts']
        concept_list = sorted(set([c[0] for c in img_concepts]))
        #for capt_id in captions:
        #  caption = ' '.join([str(cluster_id) for cluster_id in captions[capt_id]])
        print(pair['index'], img_id)

        feat_id = None
        for cur_feat_id in self.captions.keys():
          cur_img_id, _, _, _ = cur_feat_id.split('_')
          if cur_img_id == img_id:
            feat_id = cur_feat_id
        print(feat_id) 
        if not feat_id:
          continue

        if self.captions[feat_id]['is_train']:
          train_trg.write('%s\n' % ' '.join(concept_list))
        else:
          test_trg.write('%s\n' % ' '.join(concept_list))
      train_trg.close()
      test_trg.close()
    else:
      trg = open(out_prefix+'_all_trg.txt', 'w')
      for img_id, pair in sorted(bnf_data_info.items(), key=lambda x:x[1]['index']):
        img_concepts = pair['image_concepts']
        concept_list = sorted(set([c[0] for c in img_concepts]))
        #for capt_id in captions:
        #  caption = ' '.join([str(cluster_id) for cluster_id in captions[capt_id]])
        print(pair['index'], img_id)

        feat_id = None
        for cur_feat_id in self.captions.keys():
          cur_img_id, _, _, _ = cur_feat_id.split('_')
          if cur_img_id == img_id:
            feat_id = cur_feat_id
        print(feat_id) 
        if not feat_id:
          continue
        
        trg.write('%s\n' % ' '.join(concept_list))


    # Generate the bottleneck feature (.npz) file 
    '''
    train_src_dir = out_prefix+'_train/'
    test_src_dir = out_prefix+'_test/'
    
    fp = open(gold_align_file, 'r')
    align_info = json.load(fp)
    fp.close()
    
    if not os.path.isdir(train_src_dir):
      os.mkdir(train_src_dir)
      os.mkdir(test_src_dir)
    
    for img_id, pair in sorted(bnf_data_info.items(), key=lambda x:x[1]['index']):
      # Find the captions that have gold alignment; do not need the outer for-loop if every caption has gold alignment
      print(img_id)
      for i_a, align in enumerate(align_info):
        if align['image_id'] == img_id:
          capt_id = align['capt_id']
          feat_id = None
          for cur_feat_id in self.captions:
            cur_img_id, _, cur_capt_id, _ = cur_feat_id.split('_') 
            if cur_img_id == img_id and cur_capt_id == capt_id:
              feat_id = cur_feat_id
              break
          if feat_id in self.bnf_train:
            np.save(train_src_dir + '_'.join([img_id, capt_id, str(i_a)])+'.npy', self.bnf_train[feat_id])
          else:
            np.save(test_src_dir + '_'.join([img_id, capt_id, str(i_a)])+'.npy', self.bnf_test[feat_id])
    '''

  # Map between the two transcripts of the same sentence since the word segmentation and the word-concept alignment are created 
  # with a set of different tokenization rules for compound noun
  def _map_transcripts(self, trg_sent, ref_sent):
    trg2ref = [[] for i in range(len(trg_sent))]
    ref2trg = [[] for i in range(len(ref_sent))]
    
    # Find all the exact match of words in the two transcriptions
    for i_t, tw in enumerate(trg_sent):
      for i_r, rw in enumerate(ref_sent):
        if tw == rw and len(ref2trg[i_r]) == 0:
          trg2ref[i_t].append(i_r)
          ref2trg[i_r].append(i_t)
          break 
    
    # Find the match for compound nouns (assume compound nouns appear in isolation)
    # TODO: handle the case when two compound nouns appear in adjacence
    for i_t, tw in enumerate(trg_sent):
      if len(trg2ref[i_t]) == 0:
        # Find the first consecutive strings of word in the ref sent
        # that does not align to any words
        st = -1
        for i_r, rw in enumerate(ref_sent):
          if len(ref2trg[i_r]) == 0:
            st = i_r
            break
        
        for i_r, rw in enumerate(ref_sent[st:]):
          if len(ref2trg[st+i_r]) > 0:
            break
          trg2ref[i_t].append(st+i_r)   
          ref2trg[st+i_r].append(i_t)
    
    return trg2ref, ref2trg

  def _merge_repeated_frames(self, cluster_ids):
    cur_id = cluster_ids[0]
    merged = [str(cur_id)]
    for cluster_id in cluster_ids:
      if cluster_id == cur_id:
        continue
      else:
        cur_id = cluster_id
        merged.append(str(cur_id))
    return ' '.join(merged) 

if __name__ == '__main__':
  data_info_file = '../data/flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json'
  train_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_train.npz'
  test_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_test.npz'
  caption_seqs_file = 'caption_seqs.json' 
  phnset = 'pseudophns.npy'
  word_align_dir = '/home/lwang114/data/flickr/word_segmentation/'
  out_file = '../data/flickr30k/audio_level/flickr_bnf_concept_info.json'
  word_concept_align_file = '../data/flickr30k/word_level/flickr30k_gold_alignment.json'
  gold_align_file = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json' 
  bn_preproc = FlickrBottleneckPreprocessor(train_file, test_file, data_info_file, word_align_dir, phone_centroids_file=phnset, caption_seqs_file=caption_seqs_file)
  #bn_preproc.extract_info(out_file=out_file)
  #bn_preproc.label_captions()
  #bn_preproc.create_gold_alignment(out_file, word_concept_align_file, out_file=gold_align_file)
  bn_preproc.json_to_xnmt_format(out_file, gold_align_file, train_test_split=False)
  #wrd_segment_loader = WordSegmentationLoader(word_align_dir)
  #wrd_segment_loader.extract_info()  wrd_segment_loader.generate_gold_audio_concept_alignment(word_concept_align_file, bn_info_file, word_align_dir)
