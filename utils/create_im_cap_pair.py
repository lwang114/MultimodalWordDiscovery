import numpy as np
import scipy.io as sio
import glob 
import json

DEBUG = False
forceali_path = '/home/lwang114/data/flickr/word_segmentation/'
im_path = '/home/lwang114/data/flickr/Flickr8k_Dataset/'
bb_path = '/home/lwang114/data/flickr/flickr30k_label_bb/bboxes.mat'
imlabel_path = '/home/lwang114/data/flickr/flickr30k_label_bb/flickr30k_phrases.txt'

# Create a dictionary to map id to audio filenames using forceali_path
'''audfilelist = glob.glob(forceali_path+'*')
with open('word_ali_list.txt', 'w') as f:
  f.write('\n'.join(audfilelist))


idToAud = {}
for f in audfilelist:
  data_id = f.split('/')[-1].split('_')[0]
  #if DEBUG:
  #  print(f)
  if not data_id in idToAud.keys():
    idToAud[data_id] = [f]
  else:
    idToAud[data_id].append(f)

with open('idToAud.json', 'w') as f:
  json.dump(idToAud, f)
'''
with open('idToAud.json', 'r') as f:
  idToAud = json.load(f)

bboxes_info = sio.loadmat(bb_path)
bboxes = bboxes_info['bboxes_arr']

# Load the image labels to a list, and for each label in the list, get the corresponding force alignment files, read through all the words in the ali file to find when the words start and end in the audio
with open(imlabel_path, 'r') as g:
   im_info = g.read().strip().split('\n')

im_capt_pairs = [] 

for i, info in enumerate(im_info):
  data_id = info.split(' ')[0].split('_')[0]
  
  ### NOTICE: Ignore the description ahead of the entity in the phrase; the last word tends to be the entity; also ignore the entity if it repeats the entity before and instead save all the bounding boxes for the same entity and taking the union of the boxes  
  trg_wrd = info.split(' ')[-2]
  '''if trg_wrd == prev_trg_wrd:
    
    flag = 1 
    bbox_grp.append()
  
  if flag = 1:
  '''

  if len(trg_wrd.strip().split('-')) > 1:
    trg_wrd = trg_wrd.strip().split('-')[-1] 
  #if DEBUG:
  print(trg_wrd)
  # Go through each utterance to find the start and end of the word; if the word id are not found in the dictionary, skip it as it may be discarded due to noise in previous dataset cleanup 
  if data_id in idToAud.keys():
    aud_files = idToAud[data_id]
  else:
    continue 
  start = '-1'
  end = '-1'
  for af in aud_files:
    with open(af, 'r') as f:
      ali = f.read().strip().split('\n')
      for line in ali:
        line_parts = line.strip().split()
        # Ignore empty, corrupted files
        if line_parts == []:
          continue
        if DEBUG:
          print(line_parts)
        wrd = line_parts[0]
        start = line_parts[1]
        end = line_parts[2]
        if wrd == trg_wrd.upper():
          break
  # Create a string containing the info for the image-caption pair 
  imf = im_path + '_'.join(af.split('/')[-1].split('_')[:-1]) + '.jpg'
  bbox = bboxes[i] 
  if DEBUG:
    print(' '.join([str(bbox[2]), str(bbox[3])]))
  im_capt_pair = ' '.join([imf, af, trg_wrd, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), start, end]) 
  im_capt_pairs.append(im_capt_pair)
  with open('flickr_im_capt_pairs.txt', 'w') as f:
    f.write('\n'.join(im_capt_pairs))
