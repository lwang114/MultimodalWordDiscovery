# Multimodal Segmental KMeans Model
# ---------------------------------
# Author: Liming Wang, 
# Part of code modified from https://github.com/kamperh/bucktsong_segmentalist

import numpy as np
import time
from copy import deepcopy
import json
import random
import scipy.signal as signal
import scipy.interpolate as interpolate 
import logging
import os
from audio_kmeans_word_discoverer import *

NULL = 'NULL'
DEBUG = False
flatten_order = "C"
if os.path.exists("*.log"):
  os.system("rm *.log")

logging.basicConfig(filename="audio_segembed_kmeans_word_discoverer.log", format="%(asctime)s %(message)s", level=logging.DEBUG)
#logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

class SegEmbedKMeansWordDiscoverer:
    def __init__(self, numMixtures, sourceCorpusFile=None, targetCorpusFile=None, 
                  embedDim=300, minWordLen=20, maxWordLen=80,
                  fCorpus=None, tCorpus=None,
                  centroidFile=None, boundaryFile=None, 
                  initMethod="rand"):
      self.fCorpus = fCorpus
      self.tCorpus = tCorpus
      if sourceCorpusFile and targetCorpusFile:
        self.parseCorpus(sourceCorpusFile, targetCorpusFile)
      
      self.embeddings = []
      self.centroids = {}
      self.assignments = []
      self.segmentations = []
      self.numMembers = {}
      
      self.numMixtures = numMixtures
      self.embedDim = embedDim
      self.minWordLen = minWordLen
      self.maxWordLen = maxWordLen

      self.avgCentroidDistance = np.inf
      self.kmeans = None
      begin_time = time.time()
      self.initialize(centroidFile=centroidFile, boundaryFile=boundaryFile, initMethod=initMethod)
      print("Takes %0.5f to finish initialization using %s" % (time.time() - begin_time, initMethod))

    # Tokenize the corpus 
    def parseCorpus(self, sourceFile, targetFile, maxLen=1000):
      fp = open(targetFile, 'r')
      tCorpus = fp.read().split('\n')
      self.tCorpus = [[NULL] + sorted(tSen.split()) for tSen in tCorpus]
      fCorpus = np.load(sourceFile)
      self.fCorpus = [fCorpus[fKey] for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
      self.fCorpus = [fSen[:maxLen] for fSen in self.fCorpus]
      self.data_ids = [fKey for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
      
      self.featDim = self.fCorpus[0].shape[1] 

    def initialize(self, centroidFile=None, boundaryFile=None, initMethod='kmeans++', p_boundary=0.5):
      # Initialize segmentations
      if boundaryFile:
        boundaries = np.load(boundaryFile)
        segmentations = []
        for b_vec in boundaries:
          segmentation = [0]
          b_prev = 0
          for b in b_vec:
            # Avoid repeated segmentation
            if b[1] == b_prev:
              print("Repeated segmentation")
              continue
            segmentation.append(b[1])
            b_prev = b[1]
          segmentations.append(segmentation)
        self.segmentations = segmentations

      else:
        # Initialize a random segmentation and then use kmeans++
        self.segmentations = []
        for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
          fLen = fSen.shape[0]
          if p_boundary > 0:
            b_vec = np.random.binomial(1, p_boundary, size=(fLen,))
            #b_vec = np.ones((fLen,))
          else:
            b_vec = np.array([fLen])
          
          b_vec[0] = 1
          b_vec = np.asarray(b_vec.tolist() + [1])
          segmentation = list(b_vec.nonzero()[0])
          self.segmentations.append(segmentation)

      for i, (tSen, fSen, segmentation) in enumerate(zip(self.tCorpus, self.fCorpus, self.segmentations)):
        embeds_i = self.embedSent(fSen, segmentation)
        self.embeddings.append(embeds_i)   
        
      # Initialize centroids and assignments
      self.kmeans = KMeansWordDiscoverer(fCorpus=self.embeddings, tCorpus=self.tCorpus, numMixtures=self.numMixtures, centroidFile=centroidFile)
      self.centroids = self.kmeans.centroids
      self.assignments = self.kmeans.assignments
      self.numMembers = self.kmeans.numMembers
          
      if DEBUG:
        print("len(self.embeddings): %d" % len(self.embeddings))
        print("len(self.assignments): ", len(self.assignments))

    def segmentStep(self, reassign=True):
      #assert self.minWordLen * self.featDim >= self.embedDim 
      numSent = len(self.fCorpus) 
      self.segmentations = [[] for i in range(numSent)]
      
      sent_order = list(range(numSent))
      random.shuffle(sent_order)
      
      new_embeddings = [None for i in range(numSent)]
      #avgCentroidDistance = 0.
      for i in sent_order: 
        fSen = self.fCorpus[i]
        tSen = self.tCorpus[i]
        
        segmentation, _ = self.segment(fSen, tSen, self.minWordLen, self.maxWordLen, reassign=reassign, sent_id=i)
        #avgCentroidDistance += 1. / (len(self.fCorpus) * fSen.shape[0]) * segmentCost 
        self.segmentations[i] = segmentation
      
        if DEBUG:
          logging.debug("processing sentence %d" % (i))
          logging.debug("src sent len %d, trg sent len %d" % (fSen.shape[0], len(tSen)))
          logging.debug("segmentation max len %d" % (np.amax(segmentation)))

    def trainUsingEM(self, maxIterations=20, kMeansStep=3, modelPrefix='', writeModel=False):
      if writeModel:
        self.printModel(modelPrefix+'model_init.txt')

      self.prev_assignments = deepcopy(self.assignments)
      self.prev_segmentations = deepcopy(self.segmentations)
      
      n_iter = 0
      
      while (n_iter < maxIterations and not self.checkConvergence(self.prev_assignments, self.assignments)) or n_iter == 0:
        print("Starting training iteration "+str(n_iter))
        begin_time = time.time()

        self.prev_assignments = deepcopy(self.assignments)
        self.prev_segmentations = deepcopy(self.segmentations)

        #self.segmentStep(reassign=True)
        print('Segment step takes %0.5f s to finish' % (time.time() - begin_time))

        begin_time = time.time()
        self.kmeans.trainUsingEM(maxIterations=kMeansStep)
        self.centroids = self.kmeans.centroids
        self.assignments = self.kmeans.assignments

        print('K-Means step takes %0.5f s to finish' % (time.time() - begin_time))
                
        if writeModel:
          self.printModel(modelPrefix+"model_iter="+str(n_iter)+".json")
        
        n_iter += 1

      if writeModel:
        self.printModel(modelPrefix+"model_final.json")

    def checkConvergence(self, prevAssigns, curAssigns):
      if DEBUG:
        print("len(prevAssigns), len(curAssigns): ", len(prevAssigns), len(curAssigns))
      assert len(prevAssigns) == len(curAssigns)
      
      for prev_assign, cur_assign in zip(prevAssigns, curAssigns):
        if DEBUG:
          print(prev_assign, cur_assign)
        if prev_assign.shape[0] != cur_assign.shape[0]:
          return 0
        elif not (prev_assign == cur_assign).all():
          return 0 
      return 1    

    def printModel(self, filename):
      with open(filename, 'w') as f:
        centroids = {tw: c.tolist() for tw, c in self.centroids.items()}
        json.dump(centroids, f)
    
    # Embed a segment into a fixed-length vector 
    def embed(self, y, frameDim=None, technique="resample"):
      assert self.embedDim % self.featDim == 0
      if frameDim:
        y = y[:, :frameDim].T
      else:
        y = y.T
        frameDim = self.featDim

      n = int(self.embedDim / frameDim)
      
      if y.shape[0] <= 1: 
        if y.shape[0] < n:
          y = np.tile(y, [2, 1])  
        else:
          return y

      if y.shape[0] <= n:
        technique = "interpolate" 
           
      #print(xLen, self.embedDim / self.featDim)
      if technique == "interpolate":
          x = np.arange(y.shape[1])
          f = interpolate.interp1d(x, y, kind="linear")
          x_new = np.linspace(0, y.shape[1] - 1, n)
          y_new = f(x_new).flatten(flatten_order) #.flatten("F")
      elif technique == "resample":
          y_new = signal.resample(y.astype("float32"), n, axis=1).flatten(flatten_order) #.flatten("F")
      elif technique == "rasanen":
          # Taken from Rasenen et al., Interspeech, 2015
          n_frames_in_multiple = int(np.floor(y.shape[1] / n)) * n
          y_new = np.mean(
              y[:, :n_frames_in_multiple].reshape((d_frame, n, -1)), axis=-1
              ).flatten(flatten_order) #.flatten("F")
      return y_new
    
    def embedSent(self, x, segmentation):
      n_words = len(segmentation) - 1
      embeddings = []
      for i_w in range(n_words):
        seg = x[segmentation[i_w]:segmentation[i_w+1]]
        if DEBUG:
          print("start, end: ", segmentation[i_w], segmentation[i_w+1])
          print("seg.shape", seg.shape)
        embeddings.append(self.embed(seg))
      
      return np.asarray(embeddings)
    
    # TODO: Use Bregman divergence other than Euclidean distance (e.g., Itakura divergence)
    def computeDist(self, x, y, square=True):
      if square:
        return np.sum((x - y)**2, axis=-1)
      else:
        return np.sqrt(np.sum((x - y)**2, axis=-1))

    # TODO: Randomly draw a sample according to a probability mass distribution
    def randomDraw(self, pmf):
      max_val = np.sum(pmf)
      rand_val = max_val * random.random()
      rand_id = 0
      tot = 0.
      while tot < rand_val:
        tot += pmf[rand_id] 
        rand_id += 1
      return rand_id         

    def assign(self, fSen, tSen, segmentation):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      tSen = sorted(tSen)
      numWords = len(segmentation) - 1
      dist_mat = np.inf * np.ones((tLen, self.numMixtures, numWords))
      assignDists = np.zeros((numWords, tLen))
      embeddings = self.embedSent(fSen, segmentation)
      #print("embeddings.shape", np.asarray(embeddings).shape)
      for i_w in range(numWords):
        for i_t, tw in enumerate(tSen):
          for m in range(self.numMixtures):
            dist_mat[i_t, m, i_w] = self.kmeans.computeDist(embeddings[i_w], self.centroids[tw][m])

      assignment = np.argmin(dist_mat.reshape(-1, numWords), axis=0)
      assignDists = dist_mat.min(axis=1).T
      return assignment, assignDists         
        
    def segment(self, fSen, tSen, minWordLen, maxWordLen, reassign=False, sent_id=None):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      tSen = sorted(tSen)
      segmentCosts = np.inf * np.ones((fLen,))
      segmentAssigns = np.nan * np.ones((fLen,)) 
      segmentPaths = [0]*fLen
      segmentCosts[0] = 0.
      segmentPaths[0] = -1
      
      embeds = np.zeros((fLen * (maxWordLen - minWordLen + 1), self.embedDim))
      embedIds = -1 * np.ones((fLen * (fLen + 1) / 2, )).astype("int")
      i_embed = 0
      for cur_end in range(minWordLen-1, fLen):
        for cur_len in range(minWordLen, maxWordLen+1):
          if cur_end - cur_len + 1 == 0 or cur_end - cur_len + 1 >= minWordLen:
            t = cur_end
            i = t * (t + 1) / 2 + cur_len - 1
            
            #if DEBUG:
            #  print("end, len: %s %s" % (str(t), str(cur_len)))
            
            embedIds[i] = i_embed
            embeds[i_embed] = self.embed(fSen[t-cur_len+1:t+1]) 
            i_embed += 1
      
      for i_f in range(minWordLen-1, fLen):
        cur_embeds = []
        cur_embedLens = []

        for j_f in range(minWordLen, maxWordLen+1):
          if i_f - j_f + 1 == 0 or i_f - j_f + 1 >= minWordLen:
            t = i_f
            i = t * (t + 1) / 2 + j_f - 1
            i_embed = embedIds[i]
            cur_embeds.append(embeds[i_embed])
            cur_embedLens.append(j_f)
            
            if DEBUG:
              logging.debug("segment ended at %d with length %d" % (t, j_f))
        
        numCandidates = len(cur_embeds)
        costs = np.zeros((tLen, self.numMixtures, numCandidates))  
        
        t = i_f
        start = (t - np.array(cur_embedLens) + 1).tolist()    
          
        for i_t, tw in enumerate(tSen):     
          for m in range(self.numMixtures):
            # Distance weighted by the number of frames
            costs[i_t, m, :] = segmentCosts[start] + np.asarray(cur_embedLens) * self.computeDist(np.asarray(cur_embeds), self.centroids[tw][m])
            # Unweighted distance
            #costs[i_t, m, :] = segmentCosts[start] + self.computeDist(np.array(cur_embeds), self.centroids[tw][m])            

        minCosts = np.amin(costs, axis=(0, 1))
        minCost = np.amin(minCosts)
        bestLen = cur_embedLens[np.argmin(minCosts)]
        segmentCosts[i_f] = segmentCosts[max(i_f - bestLen, 0)] + minCost
        segmentAssigns[i_f] = np.argmin(np.amin(costs, axis=(1, 2)))
         
        if i_f - bestLen >= 0:
          segmentPaths[i_f] = i_f - bestLen
        else:
          segmentPaths[i_f] = 0
      
        if DEBUG:
          logging.debug("start time of the current segments: " + str(start))
          logging.debug("end time of the current segments: " + str(i_f))
          logging.debug("len(cur_embeds): " + str(len(cur_embeds)))
          logging.debug("Costs: " + str(minCosts))
          logging.debug("best segmentation point: " + str(cur_embedLens[np.argmin(minCosts)])) 

      # Follow the back pointers to find the optimal segmentation
      i_f = fLen - 1
      best_segmentation = []
      new_embeds = []
      new_assigns = [] 
      while segmentPaths[i_f] >= 0:
        i_f_prev = segmentPaths[i_f]
        embed_len = i_f - i_f_prev
        
        i = i_f * (i_f + 1) / 2 + embed_len - 1
        embed_id = embedIds[i]

        new_embeds.append(embeds[embed_id])
        new_assigns.append(segmentAssigns[i_f])
        best_segmentation.append(i_f)
        i_f = i_f_prev 

      if DEBUG:
        logging.debug("segment costs: %s" % str(segmentCosts))
        #logging.debug("best segment cost: %s" % str(segmentCosts[fLen - 1]))
        logging.debug("best segment path: %s" % str(best_segmentation[::-1]))

      best_segmentation = best_segmentation[::-1]
      new_embeds = np.asarray(new_embeds[::-1])
      new_assigns = np.asarray(new_assigns[::-1])
      assert len(new_embeds) == len(best_segmentation)

      # Update the centroids based on the new assignments
      if reassign:
        assert sent_id is not None
        old_segments = self.segmentations[sent_id] 
        old_embeds = self.embedSent(fSen, old_segments)
        self.kmeans.reassign(sent_id, new_embeds, new_assigns)

      return best_segmentation, segmentCosts[-1]

    def align(self, fSen, tSen, sent_id=None):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      tSen = sorted(tSen)
      alignment = [0]*fLen
      align_probs = -np.inf * np.ones((fLen, tLen))
      
      if sent_id:
        segmentation = self.segmentations[sent_id]
      else:
        segmentation, segmentCost = self.segment(fSen, tSen, self.minWordLen, self.maxWordLen, reassign=False)
      
      assignment, assignDists = self.assign(fSen, tSen, segmentation)
      numWords = len(segmentation)-1

      for i_w in range(numWords):
        start, end = segmentation[i_w], segmentation[i_w+1]
        alignment[start:end] = [int(assignment[i_w] / self.numMixtures)] * (end - start) 
        align_probs[start:end] = -assignDists[i_w]
      return alignment, align_probs

    def printAlignment(self, filePrefix):
      f = open(filePrefix+'.txt', 'w')
      aligns = []
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment, align_probs = self.align(fSen, tSen, sent_id=i)
        align_info = {
          'index': self.data_ids[i],
          'image_concepts': tSen,
          'alignment': alignment,
          'align_probs': align_probs.tolist(),
          'is_phoneme': False,
          'is_audio': True
          } 
        aligns.append(align_info)

        f.write('%s\n%s\n' % (tSen, fSen))
        for a in alignment:
          f.write('%d ' % a)
        f.write('\n\n')

      f.close()
    
      # Write to a .json file for evaluation
      with open(filePrefix+'.json', 'w') as f:
        json.dump(aligns, f, indent=4, sort_keys=True)            

if __name__ == '__main__':
  #datapath = '../data/flickr30k/audio_level/'
  #src_file = 'flickr_bnf_all_src.npz'
  #trg_file = 'flickr_bnf_all_trg.txt'
  datapath = "./"
  src_file = "small.npz"
  trg_file = "small.txt"
  boundary_file = datapath+"small_boundary.npy"

  model = SegEmbedKMeansWordDiscoverer(1, datapath+src_file, datapath+trg_file, boundaryFile=boundary_file, embedDim=390, minWordLen=10, maxWordLen=40)
  model.trainUsingEM(maxIterations=100, kMeansStep=3, writeModel=True, modelPrefix="random_kmeans_embed_")
  model.printAlignment("random_kmeans_embed_pred")
