# Multimodal Segmental Gaussian Mixture Model
# ---------------------------------
# Author: Liming Wang, 
# Part of the code modified from https://github.com/kamperh/bucktsong_segmentalist

import numpy as np
import time
from copy import deepcopy
import json
import random
import scipy.signal as signal
import scipy.interpolate as interpolate
import logging
import os
#from _cython_utils import *
from audio_gmm_word_discoverer import *
from audio_kmeans_word_discoverer import * 

NULL = 'NULL'
EPS = np.finfo(float).eps
DEBUG = False
flatten_order = "C"
if os.path.exists("*.log"):
  os.system("rm *.log")

# XXX
#random.seed(2)
#np.random.seed(2)
logging.basicConfig(filename="audio_segembed_gmm_word_discoverer.log", format="%(asctime)s %(message)s", level=logging.DEBUG)
#logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

class SegEmbedGMMWordDiscoverer:
    def __init__(self, numMixtures, frameDim, 
              sourceCorpusFile=None, targetCorpusFile=None, 
              landmarkFile=None, 
              modelDir=None,
              fCorpus=None, tCorpus=None,
              embedDim=None, minWordLen=20, maxWordLen=80):
      self.fCorpus = fCorpus
      self.tCorpus = tCorpus
      if sourceCorpusFile and targetCorpusFile:
        self.parseCorpus(sourceCorpusFile, targetCorpusFile)
      self.frameDim = frameDim  
      self.centroids = {}
      self.assignments = []
      self.segmentations = []
      self.embeddings = []
      self.numMembers = {}
      self.numMixturesMax = numMixtures
      self.numMixtures = {}
      
      self.embedDim = embedDim
      self.featDim = self.fCorpus[0].shape[1]
      #self.minWordLen = minWordLen
      #self.maxWordLen = maxWordLen
      self.logProbX = -np.inf  
     
      self.mixturePriorFile = None
      self.transMeanFile = None
      self.transVarFile = None 
      if modelDir:
        self.mixturePriorFile = modelDir + "model_final_mixture_priors.json"
        self.transMeanFile = modelDir + "model_final_translation_means.json"
        self.transVarFile = modelDir + "model_final_translation_variances.json" 
      self.initialize(landmarkFile, mixturePriorFile=self.mixturePriorFile, transMeanFile=self.transMeanFile, transVarFile=self.transVarFile, initMethod="rand")

    # Tokenize the corpus 
    def parseCorpus(self, sourceFile, targetFile, maxLen=2000):
      fp = open(targetFile, 'r')
      tCorpus = fp.read().split('\n')
      # XXX XXX
      self.tCorpus = [[NULL] + sorted(tSen.split()) for tSen in tCorpus]
      fCorpus = np.load(sourceFile)
      # XXX XXX
      self.fCorpus = [fCorpus[fKey] for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
      self.fCorpus = [fSen[:maxLen] for fSen in self.fCorpus] 
      self.data_ids = [fKey for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]

    def initialize(self, landmarkFile=None, mixturePriorFile=None, transMeanFile=None, transVarFile=None, initMethod='kmeans++', p_boundary=0.5):
      if transMeanFile:
        with open(transMeanFile, 'r') as f:
          self.transMeans = json.load(f)

        with open(transVarFile, 'r') as f:
          self.transVars = json.load(f)
        
        with open(mixturePriorFile, 'r') as f:
          self.mixturePrior = json.load(f)

        self.transMeans = {tw: np.array(c) for tw, c in self.transMeans.items()}
        self.transVars = {tw: np.array(v) for tw, v in self.transVars.items()}
        self.mixturePriors = {tw: np.array(p) for tw, p in self.mixturePrior.items()}
        self.numMixtures = {tw: m.shape[0] for tw, m in self.transMeans.items()}
        return
      
      if landmarkFile:
        landmarks = np.load(landmarkFile)
        #XXX XXX
        for lm_id in sorted(landmarks, key=lambda x:int(x.split('_')[-1])):
          segmentation = []
          for b in landmarks[lm_id]:
            segmentation.append(b)
          self.segmentations.append(segmentation)
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
      
      for i, (fSen, segmentation) in enumerate(zip(self.fCorpus, self.segmentations)):      
        if DEBUG:
          print(i)
        self.embeddings.append(self.getSentEmbeds(fSen, segmentation, frameDim=self.frameDim))
      
      if DEBUG:
        print("embeddings[0].shape: ", self.embeddings[0].shape)
        print("embeddings[0][:10]", self.embeddings[0][:10])
      self.acoustic_model = GMMWordDiscoverer(
                    fCorpus=self.embeddings, tCorpus=self.tCorpus,
                    numMixtures=self.numMixturesMax, 
                    mixturePriorFile=mixturePriorFile, 
                    transMeanFile=transMeanFile, 
                    transVarFile=transVarFile, 
                    initMethod=initMethod
                    )

    def trainUsingEM(self, numIterations=10, numGMMSteps=1, centroidFile=None, modelPrefix='', writeModel=False, initMethod='kmeans++'):
      if writeModel:
        self.printModel(modelPrefix+'model_init.txt')

      self.prev_segmentations = deepcopy(self.segmentations)
      
      n_iter = 0
      prevLogProbX = -np.inf
      for n_iter in range(numIterations): 
        print("Starting training iteration "+str(n_iter))
        begin_time = time.time()
        
        #self.segmentStep()
        #print('Segment step takes %0.5f s to finish' % (time.time() - begin_time))

        begin_time = time.time()
        self.acoustic_model.trainUsingEM(numIterations=numGMMSteps)
        print('GMM training takes %0.5f s to finish' % (time.time() - begin_time))
 
        if writeModel:
          self.printModel(modelPrefix+"model_iter="+str(n_iter))
         
      if writeModel:
        self.printModel(modelPrefix+"model_final")
         
    def printModel(self, filename):
      self.acoustic_model.printModel(filename) 
    
    # Embed a segment into a fixed-length vector 
    def embed(self, y, frameDim=None, technique="resample"):
      #assert self.embedDim % self.featDim == 0
      if frameDim:
        y = y[:, :frameDim].T
      else:
        y = y.T
        frameDim = self.featDim

      n = int(self.embedDim / frameDim)
      if y.shape[0] == 1: 
        y_new = np.repeat(y, n)   
  
      #if y.shape[0] <= n:
      #  technique = "interpolate" 
           
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
    
    def getSentEmbeds(self, x, segmentation, frameDim=12):
      n_words = len(segmentation) - 1
      embeddings = []
      for i_w in range(n_words):
        seg = x[segmentation[i_w]:segmentation[i_w+1]]
        if DEBUG:
          print("seg.shape", seg.shape)
          print("seg:", segmentation[i_w+1])
          print("embed of seg:", self.embed(seg))
        embeddings.append(self.embed(seg, frameDim=frameDim))  
      return np.array(embeddings)
     
    def getSentDurations(self, segmentation):
      n_words = len(segmentation) - 1
      durations = []
      for i_w in range(n_words):
        durations.append(segmentation[i_w+1]-segmentation[i_w])
      return durations

    def getAssignScores(self, embedding, tSen):
      return [gmmProb(embedding, 
                  self.acoustic_model.mixturePriors[tw],
                  self.acoustic_model.transMeans[tw],
                  self.acoustic_model.transVars[tw], log_prob=True) 
              for tw in tSen]

    def getSentAssignScores(self, embeddings, tSen):
      return [self.getAssignScores(embedding, tSen) for _, embedding in enumerate(embeddings)]

    def assign(self, i):
      embeddings = self.embeddings[i]
      segmentation = self.segmentations[i]
      assignScores = self.getSentAssignScores(embeddings, self.tCorpus[i])
      assignment = np.argmax(np.array(assignScores), axis=1).tolist()
      
      return assignment, assignScores        
    
    def align(self, i):
      fSen = self.fCorpus[i]
      fLen = fSen.shape[0]
      tSen = sorted(self.tCorpus[i])
      tLen = len(self.tCorpus[i])
      alignment = []
      align_probs = []
      embeddings = self.embeddings[i]
      segmentation = self.segmentations[i]
      embeddings = self.getSentEmbeds(fSen, segmentation) 
      durations = self.getSentDurations(segmentation)

      assignment, assign_scores = self.assign(i)    
      start = 0
      for j, scores, dur in zip(assignment, assign_scores, durations):  
        alignment.extend([j] * dur)
        align_probs.extend([scores] * dur)
        start = start + dur
      return alignment, align_probs
    
    def printAlignment(self, filePrefix):
      f = open(filePrefix+'.txt', 'w')
      aligns = []
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment, align_probs = self.align(i)
        align_info = {
          'index': self.data_ids[i],
          'image_concepts': tSen,
          'alignment': alignment,
          'align_probs': align_probs,
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

# TODO
'''def segmentStep(self):
      assert self.minWordLen * self.featDim >= self.embedDim 
      self.segmentations = []
      numSent = len(self.fCorpus) 
      sent_order = list(range(numSent))
      random.shuffle(sent_order)
       
      for i in sent_order: 
        fSen = self.fCorpus[i]
        tSen = self.tCorpus[i]
        #if DEBUG:
        logging.debug("processing sentence %d" % (i))
        logging.debug("src sent len %d, trg sent len %d" % (fSen.shape[0], len(tSen)))

        segmentation, segmentProb = self.segment(fSen, tSen, self.minWordLen, self.maxWordLen, reassign=True)        
        self.segmentations.append(segmentation)
        self.logProbX += 1. / numSent * segmentProb
'''

'''
    def checkConvergence(self, curLikelihood, prevLikelihood, tol=1e-3):
      if prevLikelihood == -np.inf:
        return False
      elif abs(curLikelihood - prevLikelihood) / abs(curLikelihood + EPS) < tol:
        return True
      else:
        return False
    '''
 
'''def reassign(self, tSen, 
                newEmbeds, oldEmbeds, 
                oldAssignProbs, newAssignProbs):
      tSen = sorted(tSen)
      newMeans = {}
      newVars = {}
      # Remove old components
      for k_t, tw in enumerate(tSen):
        for m in range(self.numMixtures[tw]):
          self.mixturePriors[tw][m] = np.exp(self.mixturePriors[tw][m]) * self.numMembers[tw][m] - np.sum(np.exp(self.oldAssignProbs[:, k_t, m])) 
          self.transVars[tw][m] = self.transMeans[tw][m] * self.numMembers[tw][m] - np.dot(np.exp(oldAssignProbs[:, k_t, m]), (np.asarray(oldEmbeds) - self.transMeans[tw][m])**2) 
          self.transMeans[tw][m] = self.transMeans[tw][m] * self.numMembers[tw][m] - np.dot(np.exp(oldAssignProbs[:, k_t, m]), oldEmbeds)
          self.numMembers[tw][m] -= np.sum(np.exp(oldAssignProbs[:, k_t, m]))

          if self.numMembers[tw][m] > 0:
            self.mixturePriors[tw][m] = np.log(self.mixturePriors[tw][m]) - np.log(self.numMembers[tw][m])
            self.transMeans[tw][m] /= self.numMembers[tw][m]
            self.transVars[tw][m] /= self.numMembers[tw][m]
            
          else:
            self.removeCluster(tw, m)
            self.numMixtures[tw] -= 1

      # Add new components
      for k_t, tw in enumerate(tSen):
        for m in range(self.numMixtures[tw]):
          transVars[tw][m] = self.transVars[tw][m] * self.numMembers[tw][m] + np.dot(np.exp(newAssignProbs[:, k_t, m]), (np.asarray(newEmbeds) - self.transMeans[tw][m]) ** 2) 
          transMeans[tw][m] = self.transMeans[tw][m] * self.numMembers[tw][m] + np.dot(np.exp(newAssignProbs[:, k_t, m]), np.asarray(newEmbeds))
          self.numMembers[tw][m] += np.sum(np.exp(newAssignProbs[:, k_t, m]))
    
          self.transMeans[tw][m] /= self.numMembers[tw][m]
          self.transVars[tw][m] /= self.numMembers[tw][m]
          
    def segment(self, fSen, tSen, minWordLen, maxWordLen, reassign=False, sent_id=None):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      tSen = sorted(tSen)
      forwardProbs = -np.inf * np.ones((fLen,))
      forwardProbs[0] = 0.
      segmentAssigns = np.nan * np.ones((fLen,)) 
      
      segmentPaths = [0]*fLen
      
      embeds = np.zeros((fLen * (maxWordLen - minWordLen + 1), self.embedDim))
      if DEBUG:
        print("embeds.shape: ", embeds.shape)
      embedIds = -1 * np.ones((fLen * (fLen + 1) / 2, )).astype("int")
      i_embed = 0
      for cur_end in range(minWordLen-1, fLen):
        for cur_len in range(minWordLen, maxWordLen+1):
          if cur_end - cur_len + 1 == 0 or cur_end - cur_len + 1 >= minWordLen:
            t = cur_end
            i = t * (t + 1) / 2 + cur_len - 1 
            #if DEBUG:
            #  logging.debug("end, len: %s %s" % (str(t), str(cur_len)))
            embedIds[i] = i_embed
            #if DEBUG:
            #  print("segment.shape", fSen[t-cur_len+1:t+1].shape)
            #  print("current embed.shape: ", self.embed(fSen[t-cur_len+1:t+1]).shape) 
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
        end = i_f
        start = (end - np.array(cur_embedLens)).tolist()    
        
        # Forward filtering
        for i_t, tw in enumerate(tSen):     
          for m in range(self.numMixtures[tw]):
            # Log probability with uniform prior over concepts
            forwardProbVec = forwardProbs[start] - np.log(tLen) + np.asarray(cur_embedLens) * gmmProb(np.asarray(cur_embeds), self.mixturePriors[tw], self.transMeans[tw], self.transVars[tw], log_prob=True)
            forwardProbs[end] = logsumexp(forwardProbVec)

            # Unweighted distance
            #costs[i_t, m, :] = segmentCosts[start] + self.computeDist(np.array(cur_embeds), self.centroids[tw][m])            
        
      # Backward sampling: randomly sample a segment according to p(q_t|x_1:t, y)
      # TODO: Use gimpel sampling
      # TODO: Add annealing temperature
      segmentation = [0]
      segmentProb = 0.
      i_f = fLen - 1
      new_embeds = []
      while i_f >= 0.:
        end = i_f
        cur_embeds = [] 
        cur_embed_lens = []
        for l in range(minWordLen, maxWordLen):
          if end - l + 1 >= minWordLen or end - l + 1 == 0:
            t = i_f
            i = t * (t + 1) / 2 + l - 1
            i_embed = embedIds[i]
            cur_embeds.append(embeds[i_embed])
            cur_embed_lens.append(l)

        start = (end - np.array(cur_embed_lens) + 1).tolist()
        #if DEBUG:
        print("end, cur_embedLens: ", end, cur_embedLens)    
        
        for tw in tSen:
          print("mixturePriors.shape, numMixtures: ", self.mixturePriors[tw].shape, self.numMixtures[tw])
          backwardProbVec = forwardProbs[start] - np.log(tLen) + np.asarray(cur_embed_lens) * gmmProb(np.asarray(cur_embeds), self.mixturePriors[tw], self.transMeans[tw], self.transVars[tw], log_prob=True)
        backwardProbVec -= logsumexp(backwardProbVec)

        lenIdx = randomDraw(np.exp(backwardProbVec))
        bestLen = cur_embed_lens[lenIdx]
        segmentProb += backwardProbVec[lenIdx]

        if DEBUG:
          logging.debug("start time of the current segments: " + str(start))
          logging.debug("end time of the current segments: " + str(i_f))
          logging.debug("len(cur_embeds): " + str(len(cur_embeds)))
          logging.debug("log prob: " + str(segmentProb))
          logging.debug("best segmentation point: " + str(cur_embed_lens[np.argmin(minCosts)])) 

        segmentation.append(i_f - bestLen)
        i = i_f * (i_f + 1) / 2 + bestLen - 1
        embed_id = embedIds[i]
        new_embeds.append(embeds[embed_id])
        i_f = i_f - bestLen  

      if DEBUG:
        logging.debug("segment costs: %s" % str(segmentCosts))
        #logging.debug("best segment cost: %s" % str(segmentCosts[fLen - 1]))
        logging.debug("sampled segment path: %s" % str(segmentation[::-1]))

      segmentation = segmentation[::-1]
      new_embeds = new_embeds[::-1]
      print("len(new_embeds), len(segmentation): ", len(new_embeds), len(segmentation))
      assert len(new_embeds) == len(segmentation) - 1

      # Update the centroids based on the new assignments
      if reassign:
        assert sent_id is not None
        old_segments = self.prev_segmentations[sent_id] 
        old_assign_probs = self.prev_assign_probs[sent_id] 
        
        # TODO: Make this more efficient; avoid recomputing the assign probs
        n_words = len(segmentation)
        new_assign_probs = -np.inf * np.ones((nWords, tLen, self.numMixturesMax))
        for i_w in range(n_words):
          for k_t, tw in enumerate(tSen):
            for m in self.numMixtures[tw]:
              new_assign_probs[i_w, k_t, m] = self.mixturePriors[tw][m] + gaussian(new_embeds[i_w], self.transMeans[tw][m], self.transVars[tw][m], log_prob=True)  
        
        for i_w in range(n_words):
          norm_factor = logsumexp(new_assign_probs[i_w].flatten(order=ORD))
          new_assign_probs[i_w] -= norm_factor

        self.reassign(tSen, new_embeds, self.embeddings[sent_id], new_assign_probs, old_assign_probs)
        self.embeddings[sent_id] = new_embeds
        self.assignProbs[sent_id] = new_assign_probs

      return segmentation, segmentProb
    '''


if __name__ == '__main__':
  
  datapath = "../data/flickr30k/audio_level/"
  src_file = "flickr_mfcc_cmvn_htk.npz" #'flickr_bnf_all_src.npz'
  trg_file = "flickr_bnf_all_trg.txt"
  landmarks_file = datapath+"flickr30k_gold_landmarks.npz"

  '''
  small = {"example_"+str(i):np.random.normal(size=(12, 30)) for i in range(3)}
  datapath = landmarks_file
  src_file = "small.npz"
  trg_file = "small.txt"
  '''

  model = SegEmbedGMMWordDiscoverer(5, 12, datapath+src_file, datapath+trg_file, landmarkFile=landmarks_file, embedDim=120)
  model.trainUsingEM(writeModel=True, modelPrefix="test_gmm_embed_")
  model.printAlignment("test_gmm_embed_pred")
