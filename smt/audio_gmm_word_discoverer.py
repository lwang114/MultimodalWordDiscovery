import math
import numpy as np
import json
from nltk.tokenize import word_tokenize
from copy import deepcopy
import time
from scipy.stats import multivariate_normal
try:
  from smt.audio_kmeans_word_discoverer import *
except:
  from audio_kmeans_word_discoverer import *

from sklearn.mixture import GaussianMixture
from _cython_utils import *

# Constant for NULL word at position zero in target sentence
NULL = "NULL"
# Minimum translation probability
PMIN = 10e-12
EPS = np.finfo(float).eps
DEBUG = False
ORD = 'C'

# Compute the log sum given a list of logits in natural base 
def logSum(logits):
  minval = np.min(logits)
  return minval + np.log(np.sum(np.exp(logits - minval), axis=0))

def logDot(log_v1, log_v2):
  assert log_v1.shape[-1] == log_v2.shape[-1]
  if len(log_v2.shape) == len(log_v1.shape) and len(log_v2.shape) <= 2:
    assert log_v1.shape[0] == log_v2.shape[0]
    return logSum(log_v1 + log_v2)
  
  elif len(log_v1.shape) == 1 and len(log_v2.shape) == 2:
    out_dim = log_v2.shape[0] 
    res = np.zeros((out_dim,))
    for i in range(out_dim):
      res[i] = logSum(log_v1 + log_v2[i])
    return res
  
  elif len(log_v2.shape) == 1 and len(log_v1.shape) == 2:
    out_dim = log_v1.shape[0] 
    res = np.zeros((out_dim,))
    for i in range(out_dim):
      res[i] = logSum(log_v1[i] + log_v2)
    return res
  else:
    raise ValueError('input has to be 1-d or 2-d array')

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

class GMMWordDiscoverer:

    def __init__(self, numMixtures, sourceCorpusFile=None, targetCorpusFile=None, 
                fCorpus=None, tCorpus=None,
                mixturePriorFile=None, 
                transMeanFile=None, 
                transVarFile=None,
                initMethod="kmeans++",
                contextWidth=0):
        self.maxNumMixtures = numMixtures
        self.numMixtures = {} 
        # Initialize data structures for storing training data
        self.fCorpus = fCorpus                   # fCorpus is a list of foreign (e.g. Spanish) sentences
        self.tCorpus = tCorpus                   # tCorpus is a list of target (e.g. English) sentences
        # Read the corpus
        if sourceCorpusFile and targetCorpusFile:
          self.initialize(sourceCorpusFile, targetCorpusFile, contextWidth);
        else:
          self.data_ids = list(range(len(self.fCorpus)))

        self.transMeans = {}                     # transMeans[e_i] is initialized with means of frames of sentence in which e_i appears (TODO: better initialization) 
        self.transVars = {}                      # transVars[e_i] is initialized similarly as transMeans; assume diagonal covariance (TODO: general covariance) 
        self.alignProb = []                     # alignProb[i][k_e_k^s][m] is a list of probabilities containing expected counts for each sentence
        self.lenProb = {}
        self.avgLogTransProb = float('-inf')
        self.featDim = self.fCorpus[0].shape[1]
        self.contextWidth = contextWidth

        # Initialize any additional data structures here (e.g. for probability model)        
        self.initializeWordTranslationDensities(mixturePriorFile=mixturePriorFile, transMeanFile=transMeanFile, transVarFile=transVarFile, initMethod=initMethod) 
    
    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
    def initialize(self, fFileName, tFileName, contextWidth, maxLen=1000):
        fp = open(tFileName, 'r')
        tCorpus = fp.read().split('\n')
        self.tCorpus = [[NULL] + tw.split() for tw in tCorpus]
        fp.close()
        
        fCorpus = np.load(fFileName) 
        self.fCorpus = [concatContext(fCorpus[k], contextWidth) for k in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
        self.fCorpus = [fSen[:maxLen] for fSen in fCorpus] 
        
        self.data_ids = [feat_id.split('_')[-1] for feat_id in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
        self.featDim = self.fCorpus[0].shape[1]
        
        return
    
    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=10, writeModel=False, modelPrefix='', epsilon=1e-5, smoothing=None):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities(smoothing=smoothing)         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        print ("Initializing using KMeans ...")
        self.numMixtures = {tw: m.shape[0] for tw, m in self.transMeans.items()} 

        #self.avgLogTransProb = self.averageTranslationProbability()
        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 # <you need to implement printModel(filename)>
        #for i in range(numIterations):
        i = 0
        while i == 0 or (not self.checkConvergence(epsilon) and i < numIterations):
            print ("Average log translation probability: ", self.avgLogTransProb)
            print ("Starting training iteration "+str(i))
            begin_time = time.time()
            # Run E-step: calculate expected counts using current set of parameters
            self.computeExpectedCounts()
            print('E-step takes %0.5f s to finish' % (time.time() - begin_time))
            
            begin_time = time.time()            
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationDensities()            # <you need to implement updateTranslationProbabilities()>
            print('M-step takes %0.5f s to finish' % (time.time() - begin_time))
            
            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel(modelPrefix+'model_iter='+str(i)+'.txt')     # <you need to implement printModel(filename)>
            i += 1
            
            begin_time = time.time()
    
    # Compute translation length probabilities q(m|n)
    def computeTranslationLengthProbabilities(self, smoothing=None):
        # Implement this method
        #pass        
        #if DEBUG:
        #  print(len(self.tCorpus))
        for ts, fs in zip(self.tCorpus, self.fCorpus):
          #fs = self.fCorpus[i]
          # len - 1 since ts contains the NULL symbol
          if len(ts)-1 not in self.lenProb.keys():
            self.lenProb[len(ts)-1] = {}
          if len(fs) not in self.lenProb[len(ts)-1].keys():
            #if DEBUG:
            #  if len(ts) == 9:
            #    print(ts, fs)
            self.lenProb[len(ts)-1][len(fs)] = 1.
          else:
            self.lenProb[len(ts)-1][len(fs)] += 1.
        
        if smoothing == 'laplace':
          tLenMax = max(list(self.lenProb.keys()))
          fLenMax = max([max(list(f.keys())) for f in list(self.lenProb.values())])
          for tLen in range(tLenMax):
            for fLen in range(fLenMax):
              if tLen not in self.lenProb:
                self.lenProb[tLen] = {}
                self.lenProb[tLen][fLen] = 1.
              elif fLen not in self.lenProb[tLen]:
                self.lenProb[tLen][fLen] = 1. 
              else:
                self.lenProb[tLen][fLen] += 1. 
        # TODO: Kneser-Ney smoothing
        
        for tl in self.lenProb.keys():
          totCount = sum(self.lenProb[tl].values())  
          for fl in self.lenProb[tl].keys():
            self.lenProb[tl][fl] = self.lenProb[tl][fl] / totCount 

    # Set initial values for the translation probabilities p(f|e)
    def initializeWordTranslationDensities(self, mixturePriorFile=None, transMeanFile=None, transVarFile=None, initMethod="kmeans++"):
        # Initialize the translation mean and variance
        self.transMeans = {}
        self.transVars = {}
        self.mixturePriors = {}
        conceptCounts = {}
        self.mkmeans = KMeansWordDiscoverer(fCorpus=self.fCorpus, tCorpus=self.tCorpus, numMixtures=self.maxNumMixtures, contextWidth=self.contextWidth, initMethod=initMethod) 

        # Initialize mixture means and priors with simple cyclic assignment of centroids
        if mixturePriorFile and transMeanFile and transVarFile:
          with open(mixturePriorFile, 'r') as f:
            self.mixturePriors = json.load(f)
            self.mixturePriors = {tw:np.array(self.mixturePriors[tw]) for tw in self.mixturePriors}
            self.numMixtures = {tw:self.mixturePriors[tw].shape[0] for tw in self.mixturePriors}

          with open(transMeanFile, 'r') as f:
            self.transMeans = json.load(f)
            self.transMeans = {tw:np.array(self.transMeans[tw]) for tw in self.transMeans}
          
          with open(transVarFile, 'r') as f:
            self.transVars = json.load(f)
            self.transVars = {tw:np.array(self.transVars[tw]) for tw in self.transVars}

        else:
          self.mkmeans.trainUsingEM()
          self.transMeans = self.mkmeans.centroids
          self.mixturePriors = {tw:np.log(np.ones((centroid.shape[0],)) / float(centroid.shape[0])) for tw, centroid in self.transMeans.items()}  
          #self.transVars = {tw:np.ones((self.maxNumMixtures, self.featDim)) for tw in self.transMeans}
          self.transVars = {tw:np.zeros((self.maxNumMixtures, self.featDim)) for tw in self.transMeans}
          
          self.numMixtures = {tw:centroid.shape[0] for tw, centroid in self.transMeans.items()}

          for i, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
            fLen = fs.shape[0]
            for k_f in range(fLen):
              k_c = self.mkmeans.assignments[i][k_f]
              k_t = int(k_c / self.maxNumMixtures)
              m = k_c % self.maxNumMixtures
              
              if ts[k_t] not in conceptCounts:
                conceptCounts[ts[k_t]] = np.zeros((self.maxNumMixtures,))
              conceptCounts[ts[k_t]][m] += 1

          for i, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
            fLen = fs.shape[0]
            for k_f in range(fLen):
              k_c = self.mkmeans.assignments[i][k_f]
              k_t = int(k_c / self.maxNumMixtures)
              m = k_c % self.maxNumMixtures
               
              self.transVars[ts[k_t]][m] += np.sum((fs - self.transMeans[ts[k_t]][m]) ** 2, axis=0) / conceptCounts[ts[k_t]][m]
          

          if DEBUG: 
            print("fs: ", fs[0, :10])
            for tw in self.transVars:
              print("tw, transVars: ", tw, self.transVars[tw][0, :10])              
              print("transMeans: ", self.transMeans[tw][0, :10])


    # Run E-step: calculate expected counts using current set of parameters
    def computeExpectedCounts(self):
        # Implement this method
        # Reset align every iteration
        self.alignProb = []
        for i, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
          align = {}
          # TODO: Double check this
          fLen = fs.shape[0]
          tLen = len(ts)
          normFactor2D = np.zeros((tLen, fLen))           
          #normFactor_ = np.zeros((tLen, fLen))           
          
          for k_t, tw in enumerate(ts):    
            if tw not in align.keys():
              align[str(k_t)+'_'+tw] = np.zeros((self.numMixtures[tw], fLen))
            
            tKey = str(k_t)+'_'+tw
            for m in range(self.numMixtures[tw]):
              align[tKey][m] = self.mixturePriors[tw][m] + gaussian(fs, self.transMeans[tw][m], np.diag(self.transVars[tw][m]), log_prob=True)
              
            #normFactor_[k_t] = np.sum(np.exp(align[tKey]), axis=0) 
            for k_f in range(fLen):
              normFactor2D[k_t][k_f] = logsumexp(align[tKey][:, k_f].copy(order='C'))
          
          #normFactor_ = np.sum(normFactor_, axis=0)
          normFactor = np.zeros((fLen,))           
          for k_f in range(fLen):
            normFactor[k_f] = logsumexp(normFactor2D[:, k_f].copy(order='C'))
          
          if DEBUG:
            print("prob: ", gaussian(fs, self.transMeans[tw][0], np.diag(self.transVars[tw][0])))
            print("prob converted from log prob: ", gaussian(fs, self.transMeans[tw][0], np.diag(self.transVars[tw][0]), log_prob=True))
            #print("norm factor: ", normFactor_)
            print("norm factor converted from log prob: ", np.exp(normFactor))
 
          
          for k_t, tw in enumerate(ts):
            tKey = str(k_t)+'_'+tw
            #align[tKey] = np.exp(align[tKey] - normFactor)
            align[tKey] = align[tKey] - normFactor

          self.alignProb.append(align)
        
        # Update the expected counts
        #pass

    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationDensities(self):
        # Implement this method
        n_sentence = min(len(self.tCorpus), len(self.fCorpus))
        self.transMeans = {}
        self.transVars = {}
        self.mixturePriors = {}

        # Sum all the expected counts across sentence for pairs of translation
        normFactorList = {}
        #normFactor_ = {}
         
        for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
          for k_t, tw in enumerate(tSen):
            if tw not in self.transMeans.keys():
              self.transMeans[tw] = np.zeros((self.numMixtures[tw], self.featDim))
              self.transVars[tw] = np.zeros((self.numMixtures[tw], self.featDim))
              self.mixturePriors[tw] = np.zeros((self.numMixtures[tw],)) 
              normFactorList[tw] = [[] for _ in range(self.numMixtures[tw])]
              #normFactor_[tw] = np.zeros((self.numMixtures,))

            fLen = fSen.shape[0]
            
            tKey = str(k_t)+'_'+tw
            for m in range(self.numMixtures[tw]):
              self.transMeans[tw][m] += np.dot(np.exp(self.alignProb[i][tKey][m]), fSen) 
              
              #normFactor_[tw][m] += np.sum(np.exp(self.alignProb[i][tKey][m]))
              normFactorList[tw][m].append(logsumexp(self.alignProb[i][tKey][m]))

        normFactor = {tw: np.zeros((self.numMixtures[tw],)) for tw in normFactorList}
        
        for tw in normFactorList:
          for m in range(self.numMixtures[tw]):
            normFactor[tw][m] = logsumexp(np.asarray(normFactorList[tw][m]))
        
        # Normalize the estimated means over all audio frames
        for tw in self.transMeans.keys():
          m = 0
          while m <= self.numMixtures[tw] - 1:
            if np.exp(normFactor[tw][m]) <= 0.:
              print("Remove the empty cluster: ", tw, m)
              self.removeCluster(tw, m)
            else:
              self.transMeans[tw][m] /= np.exp(normFactor[tw][m])  
              m += 1

        # Update translation variance
        for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
          for k_t, tw in enumerate(tSen):
            tKey = str(k_t)+'_'+tw
            for m in range(self.numMixtures[tw]):
              self.transVars[tw][m] += np.dot(np.exp(self.alignProb[i][tKey][m]), ((fSen - self.transMeans[tw][m]) ** 2))
        
        # Normalization over all the possible translation of the target word
        for tw in self.transMeans.keys():
          m = 0
          while m <= self.numMixtures[tw] - 1:
            self.transVars[tw][m] /= np.exp(normFactor[tw][m])  
              
            if np.min(self.transVars[tw][m]) <= 0:
              print('bad variance, remove the cluster: ', tw, m, np.min(self.transVars[tw][m]))
              self.removeCluster(tw, m)
            else:
              self.mixturePriors[tw][m] = normFactor[tw][m] - logsumexp(normFactor[tw]) 
              m += 1

            #if np.min(self.transVars[tw][m]) <= 0:
            #  self.transVars[tw][m] = EPS

    def removeCluster(self, tw, m):
      transMeans = np.zeros((self.numMixtures[tw] - 1, self.featDim))
      transVars = np.zeros((self.numMixtures[tw] - 1, self.featDim))
      mixturePriors = np.zeros((self.numMixtures[tw] - 1,))

      new_m_id = 0
      for m_id in range(self.numMixtures[tw]):
        if m_id == m:
          continue
        else:
          transMeans[new_m_id] = self.transMeans[tw][m_id]
          transVars[new_m_id] = self.transVars[tw][m_id]
          mixturePriors[new_m_id] = self.mixturePriors[tw][m_id]
          new_m_id += 1

      self.mixturePriors[tw] = mixturePriors.copy()
      self.transMeans[tw] = transMeans.copy()
      self.transVars[tw] = transVars.copy()

      self.numMixtures[tw] -= 1


    # Compute baum anxiliary function
    def avgLogLikelihood(self, lower_bound=False):
      avgTransProb = 0.  
      avgTransProb_ = 0. 
      for i, (fs, ts) in enumerate(zip(self.fCorpus, self.tCorpus)):
        if DEBUG:
          print(ts, fs.shape)
        
        fLen = fs.shape[0]
        avgTransProb += 1. / len(self.fCorpus) * math.log(self.lenProb[len(ts)-1][fLen]) - fLen * math.log(len(ts))
        avgTransProb_ += 1. / len(self.fCorpus) * math.log(self.lenProb[len(ts)-1][fLen]) - fLen * math.log(len(ts))
        
        for k_t, tw in enumerate(ts):
          #if lower_bound:
          for m in range(self.numMixtures[tw]):
            if np.amin(self.transVars[tw][m]) <= 0:
              print("tw, mixture %d has zero variance: " % (tw, m))
          
            tKey = str(k_t)+'_'+tw
            avgTransProb_ += 1. / len(self.fCorpus) * np.dot(np.exp(self.alignProb[i][tKey][m]), -math.log(len(ts)) + self.mixturePriors[tw][m] + gaussian(fs, self.transMeans[tw][m], np.diag(self.transVars[tw][m]), log_prob=True))
          #else:
          if DEBUG:
            print(self.transMeans[tw].shape)

        for k_f in range(fLen):
          avgTransProb += 1. / len(self.fCorpus) * gmmProb(fs[k_f], self.mixturePriors[tw], self.transMeans[tw], self.transVars[tw], log_prob=True)            

      print("avgTransProb lower bound: ", avgTransProb_)
      print("avgTransProb: ", avgTransProb)
      
      return avgTransProb
      
    def checkConvergence(self, eps=1e-5):
      if len(self.alignProb) == 0:
        return 0
      avgLogTransProb = self.avgLogLikelihood()
      if self.avgLogTransProb == float('-inf'):
        self.avgLogTransProb = avgLogTransProb
        return 0
      if abs((self.avgLogTransProb - avgLogTransProb) / avgLogTransProb) < eps:
        self.avgLogTransProb = avgLogTransProb
        return 1
      
      self.avgLogTransProb = avgLogTransProb  
      return 0
    
    # Returns the best alignment between fSen and tSen using Viterbi algorithm
    def align(self, fSen, tSen):
        fLen = fSen.shape[0]
        alignment = [0]*fLen
        alignProbs = []
        for k_f in range(fLen):
          alignProb = []
          bestScore = float('-inf')
          for k_t, tw in enumerate(tSen):
            if tw not in self.mixturePriors:
              score = np.log(PMIN)
            else:
              score = gmmProb(fSen[k_f], self.mixturePriors[tw], self.transMeans[tw], self.transVars[tw], log_prob=True)
            alignProb.append(score)

            if score > bestScore:
              alignment[k_f] = k_t
              bestScore = score
          alignProbs.append(alignProb)

        return alignment, alignProbs   # Your code above should return the correct alignment instead

    # Return q(tLength | fLength), the probability of producing an English sentence of length tLength given a non-English sentence of length fLength
    # (Can either return log probability or regular probability)
    def getTranslationLengthProbability(self, fLength, tLength):
        # Implement this method
        if tLength in self.lenProb.keys():
          if fLength in self.lenProb[tLength].keys():
            return math.exp(self.lenProb[tLength][fLength])
          else:
            return float('-inf')
        else:
          return float('-inf')

    # Write this model's probability distributions to file
    def printModel(self, filename):
        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        translateMeanFile = open(filename+'_translation_means.json', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        translateVarFile = open(filename+'_translation_variances.json', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        mixturePriorFile = open(filename+'_mixture_priors.json', 'w')
        numMixtureFile = open(filename+'_num_mixtures.json', 'w')

        # TODO: Make this more memory-efficient
        transMeans = {tw: self.transMeans[tw].tolist() for tw in sorted(self.transMeans.keys())}
        transVars = {tw: self.transVars[tw].tolist() for tw in sorted(self.transVars.keys())}
        mixPriors = {tw: self.mixturePriors[tw].tolist() for tw in sorted(self.mixturePriors.keys())}
        
        json.dump(transMeans, translateMeanFile, indent=4, sort_keys=True)
        json.dump(transVars, translateVarFile, indent=4, sort_keys=True)
        json.dump(mixPriors, mixturePriorFile, indent=4, sort_keys=True)
        json.dump(self.numMixtures, numMixtureFile, indent=4, sort_keys=True)
        for tLen in self.lenProb.keys():
          for fLen in self.lenProb[tLen].keys():
            lengthFile.write('{}\t{}\t{}\n'.format(tLen, fLen, self.lenProb[tLen][fLen]))
                
        lengthFile.close()
        translateMeanFile.close()
        translateVarFile.close()
        mixturePriorFile.close()
        numMixtureFile.close()

        '''with open(filename+'_align_prob.txt', 'w') as f:
          for i, p_ali in enumerate(self.alignProb):
            p_ali_r = []
            for tw in sorted(p_ali.keys()):
              for l in range(p_ali[tw].shape[1]):
                if len(p_ali_r) <= l:
                  p_ali_r.append({tw: p_ali[tw][:, l]})
                else:
                  p_ali_r[l][tw] = p_ali[tw][:, l]
            for fr_p_ali in p_ali_r:
              for tw in sorted(fr_p_ali.keys()):
                for m in range(self.numMixtures):
                  f.write('%s %d %.5f\n' % (tw, m, fr_p_ali[tw][m]))
        ''' 

    # Write the predicted alignment to file
    def printAlignment(self, out_file_prefix, src_file=None, trg_file=None):
      if src_file and trg_file:
        self.initialize(src_file, trg_file)
      
      f = open(out_file_prefix+'.txt', 'w')
      aligns = []
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment, alignProbs = self.align(fSen, tSen)
        align_info = {
            'index': self.data_ids[i],
            'image_concepts': tSen, 
            'alignment': alignment,
            'align_probs': alignProbs,
            'is_phoneme': False,
            'is_audio': True
          }
        aligns.append(align_info)
        # TODO: Use timestamp information to make the audio sentences more readable 
        f.write('%s\n%s\n' % (tSen, fSen))
        for a in alignment:
          f.write('%d ' % a)
        f.write('\n\n')

      f.close()
    
      # Write to a .json file for evaluation
      with open(out_file_prefix+'.json', 'w') as f:
        json.dump(aligns, f, indent=4, sort_keys=True)            



# utility method to pretty-print an alignment
# You don't have to modify this function unless you don't think it's that pretty...
def prettyAlignment(fSen, tSen, alignment):
    pretty = ''
    for j in range(len(fSen)):
        pretty += str(j)+'  '+fSen[j].ljust(20)+'==>    '+tSen[alignment[j]]+'\n';
    return pretty

if __name__ == "__main__":
    # TEST 1: Compare with scikit-learn
    #datapath = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/'
    #featfile = 'flickr_40k_speech_train.npz'
    datapath = "./"
    src_file = "../data/flickr30k/audio_level/flickr_mfcc_cmvn.npz" 
    trg_file = "../data/flickr30k/audio_level/flickr_bnf_all_trg.txt" 
    boundary_file = "../data/flickr30k/audio_level/flickr30k_gold_segmentation_mfcc.npy"
  
    mixturePriorFile = "exp/july_6_gmm_mixture=5_width=2_mfcc/model_iter=2.txt_mixture_priors.json"
    transMeanFile = "exp/july_6_gmm_mixture=5_width=2_mfcc/model_iter=2.txt_translation_means.json"
    transVarFile = "exp/july_6_gmm_mixture=5_width=2_mfcc/model_iter=2.txt_translation_variances.json"

    feats = np.load(datapath + src_file)
    feat_keys = sorted(feats.keys(), key=lambda x:int(x.split('_')[-1]))[0:5]
    #print(feat_keys)
    feat_list = [feats[k] for k in feat_keys] 
    np.savez("small.npz", *feat_list) 
    X = np.concatenate(feat_list) 
    Xcontext = X #concatContext(X, 0) 

    with open("small.txt", "w") as f:
      f.write("1 2 3 4\n1 2 3 4\n1 2 3 4\n1 2 3 4\n1 2 3 4")
    
    boundaries = np.load(boundary_file)
    boundaries_subset = boundaries[:5]
    np.save("small_boundary.npy", boundaries_subset)
    
    model = GMMWordDiscoverer(1, 'small.npz', 'small.txt', contextWidth=0)
    #model = GMMWordDiscoverer(src_file, trg_file, 10, contextWidth=2)
    #model.initializeWordTranslationDensities(mixturePriorFile, transMeanFile, transVarFile)
    #model.avgLogLikelihood()
    model.trainUsingEM(numIterations=10, writeModel=True)
    
    alignment, _ = model.align(Xcontext, ["NULL", "1", "2", "3", "4"])
    print("alignment by my model: ", np.array(alignment)) 
    #print("means for my model: ", model.transMeans.values())
    
    model2 = GaussianMixture(5, covariance_type="diag")
    labels = model2.fit_predict(X)
    print("alignment by sklearn model: ", labels)
    #print("means for sklearn model: ", model2.means_)

    '''
    # TEST 2: Random examples
    datapath = '../data/flickr30k/audio_level/'
    src_file = 'flickr_bnf_all_src.npz'
    trg_file = 'flickr_bnf_all_trg.txt'
    
    datapath = "../data/random/"
    src_file = "random.npz"
    trg_file = "random.txt" 
    model = GMMWordDiscoverer(datapath+src_file, datapath+trg_file, 1)
    model.trainUsingEM(writeModel=True)
        
    # TEST 3: log probabilities  
    print(np.exp(logSum([math.log(3), math.log(2)])))
    print(np.exp(logDot(np.array([math.log(3)]), np.array([math.log(2)]))))
    print(np.exp(logDot(np.array([[math.log(3), math.log(2)], [math.log(5), math.log(1)]]), np.array([[math.log(2), math.log(6)], [math.log(4), math.log(7)]]))))
    '''