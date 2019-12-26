import numpy as np
import json
import os
import time
from sklearn.cluster import MiniBatchKMeans
from nltk.corpus import cmudict
import scipy.io.wavfile as wavfile

DEBUG = False
NONWORD = ['$']
NULL = 'NULL'
PUNCT = ['.', ',', '?', '!', '`', '\'', ';']
UNK = 'UNK'
DELIMITER = '('
FSAMPLE = 16000

class FlickrAudioPreprocessor:
  def __init__(self, bnf_train_file, bnf_test_file, info_file, word_align_dir, phone_centroids_file=None, n_clusters=60, batch_size=100, max_feature_length=1000):
    #caption_seqs_file=None
    self.bnf_train = np.load(bnf_train_file)
    self.bnf_test = np.load(bnf_test_file)
    self.pronun_dict = cmudict.dict() 
    self.word_align_dir = word_align_dir
    self.max_feat_len = max_feature_length 
    
    # Load the .npz file 
    with open(info_file, 'r') as f:
      self.data_info = json.load(f)
    
    self.captions_with_alignment = os.listdir(word_align_dir)
 
  def extract_info(self, out_file='feature_info.json'):
    bnf_data_info = {}
    # Loop through all the files in the word-level metainfo data structure
    for i, pair in enumerate(sorted(self.data_info, key=lambda x:x['index'])):
      print('Index: ', pair['index'])      
      img_filename = pair['image_filename'] 
      img_id = pair['image_id']
      # Create a new entry for the new json info file
      # Find the image id, image filename, caption text, caption phonemes, image concepts from the word-level metainfo
      # TODO: Make the file more space-efficient
      if img_id not in bnf_data_info:
        bnf_data_info[img_id] = {}
        bnf_data_info[img_id]['image_filename'] = img_filename
        bnf_data_info[img_id]['index'] = pair['index']
        bnf_data_info[img_id]['image_concepts'] = pair['image_concepts']
        bnf_data_info[img_id]['caption_phonemes'] = {}  
        bnf_data_info[img_id]['caption_texts'] = {}  
        #bnf_data_info[img_id]['caption_sequences'] = {} 

      for ali_f in self.captions_with_alignment:
        feat_id = ali_f.split('.')[0]
        cur_img_id, _, capt_id = feat_id.split('_') 
        if cur_img_id == img_id: 
          bnf_data_info[img_id]['caption_texts'][capt_id] = []
          bnf_data_info[img_id]['caption_phonemes'][capt_id] = []
          bnf_data_info[img_id]['is_train'] = caption_info['is_train']    
          ali_f = feat_id 
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
  
  # TODO: create frame-level gold alignment from word-level gold alignment and word segmentation
  def create_gold_alignment(self, audio_dir, word_segment_file, word_concept_align_file, downsample_rate=160, out_file='gold_alignment.json'):
    begin = time.time()
    audio_concept_alignments = []
    feat2wavs = {}

    with open(word_segment_file, 'r') as f:
      segment_info = json.load(f)

    with open(word_concept_align_file, 'r') as f:
      word_concept_align_info = json.load(f)

    i_a = 0
    # Loop through each caption-concept pair in the word-level info database
    for info in sorted(segment_info.values(), key=lambda x: x['index']):
      idx = info['index']
      img_id = info['image_filename'].split('.')[0]
      word_concept_align = word_concept_align_info[idx]['alignment']
      img_concepts = word_concept_align_info[idx]['image_concepts']
      concept_order = {c:i for i, c in enumerate(sorted(img_concepts))}
      sent = [w.lower() for w in word_concept_align_info[idx]['caption_text'][:-1] if len(w) > 1 or (w[0].lower() >= 'a' and w[0].lower() <= 'z')]
      #if img_id not in word_concept_align_info:
      #  print('Image id %s not found' % img_id)
      #  continue
       
      # Find the corresponding word alignment dictionary by matching the words with the image concepts
      # TODO: deal with inexact match (flickr30k has only one caption per image while flickr8k has five per image)
      word_aligns = info['caption_texts']
      concepts = [c[0] for c in info['image_concepts']]

      word_align = None
      best_IoU = 0.
      best_capt_id = None

      for capt_id, cur_word_align in word_aligns.items(): 
        cur_sent = [a[0].lower() for a in cur_word_align]
        
        cur_IoU = float(len(set(cur_sent).intersection(set(sent)))) / float(len(set(cur_sent).union(set(sent))))
        if cur_IoU > best_IoU:
          best_IoU = cur_IoU
          best_capt_id = capt_id
         
      if not best_capt_id:
        #print('caption not found')
        continue
      
      word_align = word_aligns[best_capt_id] 
      cur_sent = [a[0].lower() for a in word_align]
      trg2ref, ref2trg = self._map_transcripts(cur_sent, sent)
       
      # Create audio frame-concept alignment; comment the line below and uncomment the subsequent
      # lines if dealing with original mbn feature format (train and test in the same npz file)
      feat = None
      fead_id = None
      #if info['is_train']:
      for cur_feat_id in self.bnf_train.keys():
        cur_img_id, _, cur_capt_id, _ = cur_feat_id.split('_')
        if cur_img_id == img_id and cur_capt_id == best_capt_id:
          feat = self.bnf_train[cur_feat_id]
          feat_id = cur_feat_id
          break
      
      #else:
      for cur_feat_id in self.bnf_test.keys():
        cur_img_id, _, cur_capt_id, _ = cur_feat_id.split('_')
        if cur_img_id == img_id and cur_capt_id == best_capt_id:
          feat = self.bnf_test[cur_feat_id]
          feat_id = cur_feat_id
          break
      
      i_a += 1
      nframes = feat.shape[0]
      
      T = word_align[-1][-1] 
      audio_concept_align = [0]*nframes 
      
      # Create a mapping between the feature frames and audio frames
      feat2wav, wav2feat = self.create_feat_to_wav_map(audio_dir, feat_id, nframes, return_wav_to_feat=True, downsample_rate=downsample_rate)
      wav_len = len(wav2feat)
      
      for pos, a in enumerate(word_align):
        word, st, end = a[0], a[1], a[2]
        st_frame, end_frame = wav2feat[int(st * FSAMPLE)], wav2feat[int(end * FSAMPLE)] 
        end_frame = min(end_frame, nframes)
        for i_f in range(st_frame, end_frame):
          # If the target sentence contains a compound word that a reference sentence doesn't have
          if len(trg2ref[pos]) > 1:
            for r_pos in trg2ref[pos]:
              audio_concept_align[i_f] = word_concept_align[r_pos]
          # If the target sentence compound word does not align to any reference word; remove this case if compound words only appear in isolation  
          elif len(trg2ref[pos]) == 0:
            continue
          else:
            cur_concept = img_concepts[word_concept_align[trg2ref[pos][0]]]
            audio_concept_align[i_f] = concept_order[cur_concept]

      print('Create %d alignments after seeing %d examples' % (i_a, idx))
      # if DEBUG:
      if i_a == 1:
        #print(len(feat2wav))
        print("idx: ", idx)
        print("nframes: ", nframes)
        #print('concept indices: ', set(audio_concept_align))
      
      # Create an entry for the current caption-concept pair
      audio_concept_info = {
          'index': idx,
          'alignment': audio_concept_align,
          'caption_text': sent,
          'image_concepts': sorted(img_concepts),
          'image_id': img_id,
          'capt_id': '_'.join(feat_id.split('_')[1:3]),
        }
      audio_concept_alignments.append(audio_concept_info)
      
      feat2wavs["%s_%d" % (img_id, idx)] = feat2wav
    
    with open(out_file, 'w') as f:
      json.dump(audio_concept_alignments, f, indent=4, sort_keys=True)
    
    with open(out_file+"_feat2wav.json", "w") as f:
      json.dump(feat2wavs, f, indent=4, sort_keys=True)
    print('Take %0.5f to generate audio-concept alignment', time.time() - begin)

  # TODO: 
  def create_gold_word_landmarks(self, bnf_data_info_file, gold_alignment_file, feat_to_wav_file, downsample_rate=160, out_file="gold_landmarks.npz", max_feat_len=2000):
    with open(gold_alignment_file, "r") as f:
      gold_alignments = json.load(f)
    
    #feat_dict = np.load(feat_file)

    with open(feat_to_wav_file, "r") as f:
      feat_to_wavs = json.load(f)
      feat_to_wavs_ids = sorted(feat_to_wavs.keys(), key=lambda x:int(x.split('_')[-1]))

    with open(bnf_data_info_file, "r") as f:
      bnf_data_info = json.load(f)

    landmarks = {}
    for i, align_info in enumerate(gold_alignments):
      img_id = align_info["image_id"]
      if i >= 1 and i <= 4:
        print("image_id: ", img_id)
      capt_id = align_info["capt_id"].split("_")[-1]
      
      for feat_id in sorted(bnf_data_info, key=lambda x:bnf_data_info[x]["index"]):
        if feat_id == img_id:  
          break

      segment_info = bnf_data_info[feat_id]["caption_texts"][capt_id]
      feat_to_wav = feat_to_wavs[feat_to_wavs_ids[i]]

      nframes = len(feat_to_wav)
      wav_len = feat_to_wav[-1][1]
      wav_to_feat = np.zeros((wav_len,))
      for i_frame in range(nframes):
        start, end = feat_to_wav[i_frame]
        wav_to_feat[start:end] = i_frame    
  
      landmark = [0]
      for segment in segment_info:
        start_second, end_second = segment[1], segment[2]
        start, end = int(start_second * FSAMPLE), int(end_second * FSAMPLE)
        start_frame, end_frame = wav_to_feat[start], wav_to_feat[end]
        if end_frame <= 0 or end_frame >= min(nframes - 1, max_feat_len):
          continue  
        landmark.append(end_frame+1)
      landmark.append(min(nframes, max_feat_len))
      landmarks[feat_to_wavs_ids[i]] = landmark

      '''        
        start_frame, end_frame = None, None
        
        for frame in range(nframes):
          if start >= feat_to_wav[frame][0] and start <= feat_to_wav[frame][1]:
            start_frame = frame
          if end >= feat_to_wav[frame][0] and end <= feat_to_wav[frame][1]:
            end_frame = frame + 1
        if start_frame is not None and end_frame is not None:
          segmentation.append((start_frame, end_frame))
        elif start_frame is not None:
          segmentation.append((start_frame, nframes))
      '''

      if i >= 1 and i <= 4:
        print("nframes, last segmentation: ", nframes, landmark[-1]) 
        print("landmark: ", landmark)

    np.savez(out_file, **landmarks)
  
  def create_segment_level_gold_alignment(self, alignment_file, segmentation_file, concept_only=True, file_prefix='flickr30k_segment_level_alignment'):
    segment_npz = np.load(segmentation_file)
    segmentations = segment_npz
    segment_ids = [k for k in sorted(segment_npz, key=lambda x:int(x.split('_')[-1]))]
    # XXX
    with open(alignment_file, 'r') as f:
      align_info = json.load(f)
    align_info = align_info
    
    segment_alignments = {}  
    for i, seg_id in enumerate(segment_ids):
      print(seg_id)
      segs = segmentations[seg_id] 
      aligns = align_info[i]['alignment']
      
      seg_aligns = []
      for start, end in zip(segs[:-1], segs[1:]):
        if len(set(aligns[start:end])) > 1:
          print('non-uniform alignment within segment: ', start, end, aligns[start:end])
        
        if concept_only and np.sum(np.asarray(aligns[start:end]) > 0) > 1./2 * (end - start):
          seg_aligns.append(max(aligns[start:end])) 
      segment_alignments[seg_id] = seg_aligns

    with open(file_prefix+'.json', 'w') as f:
      json.dump(segment_alignments, f) 

  # TODO: Multiple captions for one image (so do not need to use the gold alignment file)
  def json_to_xnmt_format(self, bnf_data_info_file, gold_align_file, out_prefix='flickr_bnf', train_test_split=False):
    bnf_data_info = None
    with open(bnf_data_info_file, 'r') as f:
      bnf_data_info = json.load(f)
    
    # Generate the bottleneck feature (.npz) file 
    train_src_dir = out_prefix+'_train/'
    test_src_dir = out_prefix+'_test/'
    
    fp = open(gold_align_file, 'r')
    align_info = json.load(fp)
    fp.close()
    
    trg = open(out_prefix+'_all_trg.txt', 'w')  

    if not os.path.isdir(train_src_dir):
      os.mkdir(train_src_dir)
      os.mkdir(test_src_dir)
    
    i_pass = 0
    feats = {}
    feats_train = {}
    feats_test = {} 
    for img_id, pair in sorted(bnf_data_info.items(), key=lambda x:x[1]['index']):
      # Find the captions that have gold alignment; do not need the outer for-loop if every caption has gold alignment
      for i_a, align in enumerate(align_info):
        if align['image_id'] == img_id:
          capt_id = align['capt_id'].split("_")[1]
          wav_id = None
          for cur_wav_id in self.captions_with_alignment:
            cur_img_id, _, cur_capt_id = cur_wav_id.split('.')[0].split('_') 
            if cur_img_id == img_id and cur_capt_id == capt_id:
              wav_id = cur_wav_id
              break
          
          print(i_pass, wav_id)    
          i_pass += 1

          img_concepts = pair['image_concepts']
          concept_list = sorted(set([c[0] for c in img_concepts]))
          trg.write('%s\n' % ' '.join(concept_list))
          
          feat_id = None
          found = False
          for cur_feat_id in self.bnf_train:
            cur_img_id, _, cur_capt_id, _ = cur_feat_id.split('_')  
            if cur_img_id == img_id and cur_capt_id == capt_id:
              feat_id = cur_feat_id
              if train_test_split:
                feats_train["%s_%d" % (feat_id, i_a)] = self.bnf_train[feat_id]
              else:
                feats["%s_%d" % (feat_id, i_a)] = self.bnf_train[feat_id]
              found = True
              break

          if not found:
            for cur_feat_id in self.bnf_test: 
              cur_img_id, _, cur_capt_id, _ = cur_feat_id.split('_')  
              if cur_img_id == img_id and cur_capt_id == capt_id:
                feat_id = cur_feat_id
                if train_test_split:
                  #np.save(test_src_dir + '_'.join([img_id, capt_id, str(i_a)])+'.npy', self.bnf_test[feat_id])
                  feats_test["%s_%d" % (feat_id, i_a)] = self.bnf_test[feat_id]
                else:
                  feats["%s_%d" % (feat_id, i_a)] = self.bnf_test[feat_id]
    
    if train_test_split:
      np.savez(out_prefix+"_train_src.npz", **feats_train)
      np.savez(out_prefix+"_test_src.npz", **feats_test)
    else:
      np.savez(out_prefix+"_all_src.npz", **feats)

  def create_feat_to_wav_map(self, audio_dir, feat_id, feat_len, return_wav_to_feat=False, downsample_rate=None):    
    # Create a mapping between the feature frames and audio frames
    audio_file = "{}{}.wav".format(audio_dir, '_'.join(feat_id.split('_')[:-1]))
    fs, y = wavfile.read(audio_file)
    wav_len = y.shape[0]
    if downsample_rate is None:
      downsample_rate = int(wav_len / feat_len)
    #print(wav_len, downsample_rate)
    feat2wav = []
    wav2feat = np.zeros((wav_len,), dtype=int)
    for i_f in range(feat_len):
      if i_f < feat_len - 1:
        feat2wav.append([i_f * downsample_rate, (i_f + 1) * downsample_rate])
        if return_wav_to_feat:
          wav2feat[i_f * downsample_rate:(i_f + 1) * downsample_rate] = i_f
      else:
        feat2wav.append([i_f * downsample_rate, wav_len])
        if return_wav_to_feat:
          wav2feat[i_f * downsample_rate:wav_len] = i_f
    
    if return_wav_to_feat: 
      return feat2wav, wav2feat
    else:
      return feat2wav

  def create_feat_to_wav_maps(self, audio_dir, alignment_file):
    with open(alignment_file, 'r') as f:
      align_info = json.load(f)
    
    new_align_info = []
    feat2wavs = {}
    for i, align in enumerate(align_info):
      img_id = align['image_id']
      capt_id = align['capt_id']
      alignment = align['alignment']
      feat_len = len(alignment)
      fead_id = None
      found = False
      for cur_feat_id in self.bnf_train.keys():
        cur_img_id, _, cur_capt_id, _ = cur_feat_id.split('_')
        if cur_img_id == img_id and cur_capt_id == capt_id:
          feat_id = cur_feat_id
          found = True
          break

      if not found:
        for cur_feat_id in self.bnf_test.keys():
          cur_img_id, _, cur_capt_id, _ = cur_feat_id.split('_')
          if cur_img_id == img_id and cur_capt_id == capt_id:
            feat_id = cur_feat_id
            found = True
            break

      if not found:
        print("id not found, potential bug")
        continue
      
      print(feat_id)
      feat2wav = self.create_feat_to_wav_map(audio_dir, feat_id, feat_len) 
      feat2wavs['_'.join([feat_id, str(i)])] = feat2wav
      #align['capt_id'] = '_'.join(feat_id.split('_')[1:-1])
      #align['feat_id'] = feat_id
      #align['feat2wav'] = feat2wav
      
      #new_align_info.append(align)
    
    #with open(alignment_file, 'w') as f:
    #  json.dump(new_align_info, f) 
    with open("%s_feat2wav.json" % (alignment_file), "w") as f:
      json.dump(feat2wavs, f, indent=4, sort_keys=True)
  
  def train_test_split(self, bnf_data_info_file, out_prefix='flickr_bnf'):
    train_trg = open(out_prefix+'_train_trg.txt', 'w')
    test_trg = open(out_prefix+'_test_trg.txt', 'w')
    with open(bnf_data_info_file, 'r') as f:
      bnf_data_info = json.load(f) 

    # Generate the image concept (.txt) file
    for img_id, pair in sorted(bnf_data_info.items(), key=lambda x:x[1]['index']):
      img_concepts = pair['image_concepts']
      concept_list = sorted(set([c[0] for c in img_concepts]))
      #for capt_id in captions:
      #  caption = ' '.join([str(cluster_id) for cluster_id in captions[capt_id]])
      print(pair['index'], img_id)

      feat_id = None
      for ali_f in self.captions_with_alignment:
        cur_feat_id = ali_f.split('.')[0]
        cur_img_id, _, _ = cur_feat_id.split('_')
        if cur_img_id == img_id:
          feat_id = cur_feat_id
      print(feat_id) 
      if not feat_id:
        continue

      found = False
      for cur_feat_id in self.bnf_train: 
        cur_img_id, _, _, _ = cur_feat_id.split('_')
        if cur_img_id == img_id:
          train_trg.write('%s\n' % ' '.join(concept_list))
          found = True
          break

      if not found:
        test_trg.write('%s\n' % ' '.join(concept_list))
    train_trg.close()
    test_trg.close()
  
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
  
  # Cleanup the data by removing features for utterances that are too long 
  def data_cleanup(self, src_file, trg_file, alignment_file, feat2wav_file, max_len=1000):
    feats = np.load(src_file)
    
    with open(trg_file, "r") as f:
      concepts = f.read().strip().split("\n")
    
    with open(alignment_file, "r") as f:
      alignments = json.load(f)

    with open(feat2wav_file, "r") as f:
      feat2wavs = json.load(f)

    bad_ids = []
    feat_ids = sorted(feats, key=lambda x: int(x.split("_")[-1]))
    feat2wav_ids = sorted(feat2wavs, key=lambda x: int(x.split("_")[-1]))
    for i, (c, feat_id, ali, feat2wav) in enumerate(zip(concepts, feat_ids, alignments, feat2wavs)):
      if len(feats[feat_id]) > max_len:
        _ = feats.pop(feat_id) 
        _ = concepts.pop(i)
        _ = alignments.pop(i)
        _ = feat2wavs.pop(feat2wavs_ids[i])
    
    np.savez("cleanup_"+src_file, **feats)
    with open("cleanup_"+trg_file, "w") as f:
      json.dump("\n".join(concepts), f, indent=4, sort_keys=True)

    with open(alignment_file, "w") as f:
      json.dump(alignments, f, indent=4, sort_keys=True)

    with open(feat2wavs, "w") as f:
      json.dump(feat2wavs, f, indent=4, sort_keys=True)

if __name__ == '__main__':
  datapath = "../data/flickr30k/audio_level/"
  data_info_file = '../data/flickr30k/phoneme_level/flickr30k_info_phoneme_concept.json'
  train_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_train.npz'
  test_file = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/flickr_40k_speech_test.npz'
  caption_seqs_file = 'caption_seqs.json' 
  phnset = 'pseudophns.npy'
  word_align_dir = '/home/lwang114/data/flickr/word_segmentation/'
  out_file = '../data/flickr30k/audio_level/flickr_bnf_concept_info.json'
  word_concept_align_file = '../data/flickr30k/word_level/flickr30k_gold_alignment.json'
  
  gold_align_file = datapath + "flickr30k_gold_alignment.json" #"flickr30k_gold_alignment.json" #
  gold_lm_file = "flickr30k_gold_landmarks_mfcc.npz"
  align_file = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json'
  segment_file = '../data/flickr30k/audio_level/flickr30k_gold_landmarks_mfcc.npz'  

  feat_to_wav_file = datapath + "flickr30k_gold_alignment.json_feat2wav.json" #"flickr30k_gold_alignment.json_feat2wav.json" #"flickr_mfcc_cmvn_htk_feat2wav.json" 
  
  audio_dir = '/home/lwang114/data/flickr_audio/wavs/'
  bn_preproc = FlickrAudioPreprocessor(train_file, test_file, data_info_file, word_align_dir, phone_centroids_file=phnset)
  #bn_preproc.extract_info(out_file=out_file)
  #bn_preproc.label_captions()
  #bn_preproc.create_gold_alignment(audio_dir, out_file, word_concept_align_file, out_file=gold_align_file)
  #bn_preproc.json_to_xnmt_format(out_file, gold_align_file)
  #bn_preproc.create_feat_to_wav_maps(audio_dir, gold_align_file)
  #bn_preproc.create_gold_word_landmarks(out_file, gold_align_file, feat_to_wav_file, out_file=gold_lm_file)
  #wrd_segment_loader = WordSegmentationLoader(word_align_dir)
  #wrd_segment_loader.extract_info()  wrd_segment_loader.generate_gold_audio_concept_alignment(word_concept_align_file, bn_info_file, word_align_dir)
  bn_preproc.create_segment_level_gold_alignment(align_file, segment_file, concept_only=True, file_prefix='flickr30k_segment_level_alignment')
   
