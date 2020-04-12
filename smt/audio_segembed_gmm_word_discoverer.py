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
import math
from scipy.special import logsumexp
#from _cython_utils import *

NULL = 'NULL'
EPS = np.finfo(float).eps
DEBUG = False
flatten_order = "C"
if os.path.exists("*.log"):
  os.system("rm *.log")

random.seed(2)
np.random.seed(2)
# TODO Remove unused arguments
class SegEmbedGMMWordDiscoverer:
    def __init__(self, acousticModel, numMixtures, frameDim, 
              sourceCorpusFile=None, targetCorpusFile=None, 
              landmarkFile=None, 
              modelDir=None,
              fCorpus=None, tCorpus=None,
              embedDim=None, minWordLen=-np.inf, maxWordLen=np.inf):
      self.acoustic_model = acousticModel
      self.fCorpus = fCorpus
      self.tCorpus = tCorpus
      if sourceCorpusFile and targetCorpusFile:
        self.parseCorpus(sourceCorpusFile, targetCorpusFile)
      self.frameDim = frameDim  
      self.centroids = {}
      self.assignments = []
      self.segmentations = []
      self.embeddings = []
      self.embeddingTable = []
      self.numMembers = {}
      self.numMixturesMax = numMixtures
      self.numMixtures = {}
      
      self.embedDim = embedDim
      self.featDim = self.fCorpus[0].shape[1]
      self.minWordLen = minWordLen
      self.maxWordLen = maxWordLen
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
      self.tCorpus = [[NULL] + sorted(tSen.split()) for tSen in tCorpus[:10]]
      fCorpus = np.load(sourceFile)
      # XXX XXX
      self.fCorpus = [fCorpus[fKey] for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))[:10]]
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
        # Initialize every frame as a segment
        self.segmentations = []
        for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
          fLen = fSen.shape[0]
          self.segmentations.append(list(range(fLen)))
      
      for i, (fSen, segmentation) in enumerate(zip(self.fCorpus, self.segmentations)):      
        embedTable = np.nan * np.ones((len(fSen)-1, len(fSen), self.embedDim))
        for t in range(1, len(fSen)):
          for s in range(t):
            embedTable[s, t] = self.embed(fSen[s:t])
        self.embeddingTable.append(embedTable)
        
        # TODO: Use embed table instead of raw feature to compute this
        self.embeddings.append(self.getSentEmbeds(fSen, segmentation))

      self.acoustic_model = self.acoustic_model(
                        fCorpus=self.embeddings, tCorpus=self.tCorpus,
                        numMixtures=self.numMixturesMax, 
                        mixturePriorFile=mixturePriorFile, 
                        transMeanFile=transMeanFile, 
                        transVarFile=transVarFile, 
                        initMethod=initMethod)

    def trainUsingEM(self, numIterations=10, numGMMSteps=1, centroidFile=None, modelPrefix='', writeModel=False, initMethod='kmeans++'):
      if writeModel:
        self.printModel(modelPrefix+'model_init.txt')

      self.prev_segmentations = deepcopy(self.segmentations)
      
      n_iter = 0
      for n_iter in range(numIterations): 
        print("Starting training iteration "+str(n_iter))       

        begin_time = time.time()
        self.acoustic_model.trainUsingEM(numIterations=numGMMSteps)
        print('GMM training takes %0.5f s to finish' % (time.time() - begin_time)) 

        begin_time = time.time()        
        self.segmentStep()
        print('Segment step takes %0.5f s to finish' % (time.time() - begin_time))
 
        if writeModel:
          self.printModel(modelPrefix+"model_iter="+str(n_iter))
         
      if writeModel:
        self.printModel(modelPrefix+"model_final")

    def segmentStep(self):
      self.segmentations = []
      numSent = len(self.fCorpus) 
      sent_order = list(range(numSent))
      random.shuffle(sent_order)
       
      for i in sent_order: 
        fSen = self.fCorpus[i]
        tSen = self.tCorpus[i]
        segmentation, segmentProb = self.segment(self.embeddingTable[i], tSen, self.minWordLen, self.maxWordLen, reassign=True)        
        self.segmentations.append(segmentation)
        self.embeddings[i] = self.getSentEmbeds(fSen, segmentation)

    def segment(self, embedTable, tSen, minWordLen, maxWordLen, reassign=False, sent_id=None):
      fLen = embedTable.shape[1]
      tLen = len(tSen)
      tSen = sorted(tSen)
      forwardProbs = -np.inf * np.ones((fLen+1,))
      forwardProbs[0] = 0.
      segmentAssigns = np.nan * np.ones((fLen+1,)) 
      segmentation = [0]*(fLen+1)
      mixturePriors = np.concatenate([self.acoustic_model.mixturePriors[tw] / len(tSen) for tw in tSen], axis=0)
      transMeans = np.concatenate([self.acoustic_model.transMeans[tw] for tw in tSen], axis=0)
      transVars = np.concatenate([self.acoustic_model.transVars[tw] for tw in tSen], axis=0)
         
      for t in range(1, fLen):
        scores = forwardProbs[:t]
        scores += (t - np.arange(t)) * gmmProb(embedTable[:t, t], mixturePriors, transMeans, transVars, log_prob=True)  
        forwardProbs[t+1] = np.max(scores)
        segmentAssigns[t+1] = np.argmax(scores)

      end = fLen
      segmentation = [fLen]
      while end != 0:
        segmentation.append(int(segmentAssigns[end])) 
        end = int(segmentAssigns[end])

      return segmentation[::-1], forwardProbs
   
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
    
    def getSentEmbeds(self, x, segmentation):
      n_words = len(segmentation) - 1
      embeddings = []
      for i_w in range(n_words):
        seg = x[segmentation[i_w]:segmentation[i_w+1]]
        embeddings.append(self.embed(seg))  
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

    def printModel(self, filename):
      self.acoustic_model.printModel(filename)  
    
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
'''      

# Return log-probability of a Gaussian distribution
def gaussian(x, mean, cov, cov_type='diag', log_prob=False):
  d = mean.shape[0]
  if cov_type == 'diag':
    assert np.min(np.diag(cov)) > 0.
    if log_prob:
      log_norm_const = float(d) / 2. * np.log(2. * math.pi) + np.sum(np.log(np.diag(cov))) / 2.
      prob = - log_norm_const - np.sum((x - mean) ** 2 / (2. * np.diag(cov)), axis=-1) 
    else:
      norm_const = np.sqrt(2. * math.pi) ** float(d)
      norm_const *= np.prod(np.sqrt(np.diag(cov))) 
      x_z = x - mean
      prob = np.exp(- np.sum(x_z ** 2 / (2. * np.diag(cov)), axis=-1)) / norm_const 
  else:
    assert np.linalg.det(cov) > 0.
    chol_cov = np.linalg.cholesky(cov)
    norm_const = np.sqrt(2. * math.pi) ** float(d)
    norm_const *= np.linalg.det(chol_cov)
    prob = np.exp(-np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), x - mean) / 2.) / norm_const

  return prob

def gmmProb(x, priors, means, covs, log_prob=False):
  #log_prob = 0.
  m = priors.shape[0]
  if len(x.shape) == 1:
    probs = np.zeros((m,))
  elif len(x.shape) == 2:
    probs = np.zeros((m, x.shape[0]))
  else:
    raise ValueError('x has to be 1-d or 2-d array')

  #log_probs = np.zeros((m,))
  for i in range(m):
    if len(covs.shape) == 2:
      #log_probs[i] = gaussian(x, means[i], np.diag(covs[i]))
      probs[i] = gaussian(x, means[i], np.diag(covs[i]), log_prob=log_prob)
      #probs[i] = multivariate_normal.pdf(x, means[i], np.diag(covs[i]))
    else:
      #log_probs[i] = gaussian(x, means[i], covs[i])
      probs[i] = gaussian(x, means[i], covs[i], log_prob=log_prob)
  
  if log_prob:
    if len(x.shape) == 1:
      return logsumexp(priors + probs)
    elif len(x.shape) == 2:
      log_probs = np.zeros((x.shape[0],))
      for t in range(x.shape[0]):
        log_probs[t] = logsumexp(priors + probs[:, t])
    else:
      raise ValueError('x has to be 1-d or 2-d array')

    return log_probs
  else:
    return np.dot(priors, probs)

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
