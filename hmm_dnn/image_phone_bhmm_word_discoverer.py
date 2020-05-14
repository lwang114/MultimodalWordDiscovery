import numpy as np
import math
import json
import time
from scipy.special import logsumexp
import random
from copy import deepcopy

NULL = "NULL"
DEBUG = False
EPS = 1e-50
random.seed(1)
np.random.seed(1)

# A word discovery model using image regions and phones
# * The transition matrix is assumed to be Toeplitz 
class ImagePhoneBigramHMMWordDiscoverer:
  def __init__(self, speechFeatureFile, imageFeatureFile, modelConfigs, modelName='image_phone_hmm_word_discoverer'):
    self.modelName = modelName 
    # Initialize data structures for storing training data
    self.aCorpus = []                   # aCorpus is a list of acoustic features

    self.vCorpus = []                   # vCorpus is a list of image posterior features (e.g. VGG softmax)
    self.hasNull = modelConfigs.get('has_null', False)
    self.nWords = modelConfigs.get('n_words', 66) 
    self.momentum = modelConfigs.get('momentum', 0.)
    self.lr = modelConfigs.get('learning_rate', 10.)
    self.normalize_vfeat = modelConfigs.get('normalize_vfeat', False) 
    self.imagePosteriorFile = modelConfigs.get('image_posterior_weights_file', None)
    self.testIndices = modelConfigs.get('test_indices', [])
    self.modelPath = modelConfigs.get('model_path', None) 

    self.init = {}
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.unigramObs = None                 # obs[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
    self.bigramObs = None
    self.avgLogTransProb = float('-inf')
     
    # Read the corpus
    self.readCorpus(speechFeatureFile, imageFeatureFile, debug=False);
    if self.modelPath:     
      self.imagePosteriorFile = self.modelPath + 'image_posterior_weights.npy'
      self.initProbFile = self.modelPath + 'initialprobs.txt'
      self.transProbFile = self.modelPath + 'transitionprobs.txt'
      self.unigramObsProbFile = self.modelPath + 'unigram_observationprobs.npy'
      self.bigramObsProbFile = self.modelPath + 'bigram_observationprobs.npy'
    else:
      self.imagePosteriorFile = None
      self.initProbFile = None
      self.transProbFile = None
      self.unigramObsProbFile = None
      self.bigramObsProbFile = None
     
  def readCorpus(self, speechFeatFile, imageFeatFile, debug=False):
    aCorpus = []
    vCorpus = []
    self.phone2idx = {}
    nTypes = 0
    nPhones = 0
    nImages = 0

    vNpz = np.load(imageFeatFile)
    # XXX
    vCorpus = [vNpz[k] for k in sorted(vNpz.keys(), key=lambda x:int(x.split('_')[-1]))]
    if self.normalize_vfeat:
      vCorpus = [(vSen.T / np.linalg.norm(vSen, ord=2, axis=-1)).T for vSen in vCorpus]  
      if debug:
        print(np.linalg.norm(vCorpus[0], ord=2, axis=-1))
    if debug:
      print(len(vCorpus))
      print(vCorpus[0].shape)
    
    self.vCorpus = vCorpus
    if self.hasNull:
      # Add a NULL concept vector
      self.vCorpus = [np.concatenate((np.zeros((1, self.imageFeatDim)), vfeat), axis=0) for vfeat in self.vCorpus]   
    self.imageFeatDim = self.vCorpus[0].shape[-1]
    
    for ex, vfeat in enumerate(self.vCorpus):
      nImages += len(vfeat)
      if vfeat.shape[-1] == 0:
        print('ex: ', ex)
        print('vfeat empty: ', vfeat.shape) 
        self.vCorpus[ex] = np.zeros((1, self.imageFeatDim))
 
    if debug:
      print('len(vCorpus): ', len(self.vCorpus))

    f = open(speechFeatFile, 'r')
    aCorpusStr = []
    for line in f:
      aSen = line.strip().split()
      aCorpusStr.append(aSen)
      for phn in aSen:
        if phn not in self.phone2idx:
          self.phone2idx[phn] = nTypes
          nTypes += 1
        nPhones += 1
    f.close()
    self.audioFeatDim = nTypes

    # XXX
    for aSenStr in aCorpusStr:
      T = len(aSenStr)
      aSen = np.zeros((T, self.audioFeatDim))
      for t, phn in enumerate(aSenStr):
        aSen[t, self.phone2idx[phn]] = 1.
      self.aCorpus.append(aSen)
           
    print('----- Corpus Summary -----')
    print('Number of examples: ', len(self.aCorpus))
    print('Number of phonetic categories: ', nTypes)
    print('Number of phones: ', nPhones)
    print('Number of objects: ', nImages)
    print("Number of word clusters: ", self.nWords)
    
  def initializeModel(self, alignments=None):
    begin_time = time.time()
    self.computeTranslationLengthProbabilities()

    # Initialize the transition probs uniformly 
    for m in self.lenProb:
      self.init[m] = 1. / m * np.ones((m,))

    for m in self.lenProb:
      self.trans[m] = 1. / m * np.ones((m, m))   
  
    #print('Num. of concepts: ', self.nWords)
    if self.initProbFile:
      f = open(self.initProbFile)
      for line in f:
        m, s, prob = line.split()
        self.init[int(m)][int(s)] = float(prob)
      f.close()

    if self.transProbFile:
      f = open(self.transProbFile)
      for line in f:
        m, cur_s, next_s, prob = line.split()
        self.trans[int(m)][int(cur_s)][int(next_s)] = float(prob)     
      f.close()

    if self.unigramObsProbFile and self.bigramObsProbFile:
      self.unigramObs = np.load(self.unigramObsProbFile)
      self.bigramObs = np.load(self.bigramObsProbFile)
    else:
      self.unigramObs = 1. / self.audioFeatDim * np.ones((self.nWords, self.audioFeatDim))
      self.bigramObs = 1. / self.audioFeatDim * np.ones((self.nWords, self.audioFeatDim, self.audioFeatDim))
   
    # XXX
    #self.W = 10.*np.eye(self.nWords) 
    if self.imagePosteriorFile:
      posteriorWeights = np.load(self.imagePosteriorFile)
      weight, bias = posteriorWeights['weight'], posteriorWeights['bias']
      
      self.W = np.concatenate([weight, bias[:, np.newaxis]], axis=1)
    else:
      self.W = 1. * np.random.normal(size=(self.nWords, self.imageFeatDim + 1))
      self.W[:, -1] = 0.  
    print("Finish initialization after %0.3f s" % (time.time() - begin_time))
  
  def trainUsingEM(self, numIterations=20, writeModel=False, warmStart=False, convergenceEpsilon=0.01, printStatus=True, debug=False):
    if not warmStart:
      self.initializeModel()
    
    if writeModel:
      self.printModel('initial_model.txt')
    
    maxLikelihood = -np.inf
    likelihoods = np.zeros((numIterations,))   
    for epoch in range(numIterations): 
      begin_time = time.time()
      initCounts = {m: np.zeros((m,)) for m in self.lenProb}
      transCounts = {m: np.zeros((m, m)) for m in self.lenProb}
      unigramCounts = np.zeros((self.nWords, self.audioFeatDim)) 
      bigramCounts = np.zeros((self.nWords, self.audioFeatDim, self.audioFeatDim))      
      conceptCounts = [np.zeros((vSen.shape[0], self.nWords)) for vSen in self.vCorpus]
      self.conceptCountsA = [np.zeros((aSen.shape[0], self.nWords)) for aSen in self.aCorpus]

      if printStatus:
        likelihood = self.computeAvgLogLikelihood()
        likelihoods[epoch] = likelihood
        print('Epoch', epoch, 'Average Log Likelihood:', likelihood)  
        if writeModel and likelihood > maxLikelihood:
          self.printModel(self.modelName + '_iter='+str(epoch)+'.txt')
          self.printAlignment(self.modelName+'_iter='+str(epoch)+'_alignment', debug=False)          
          maxLikelihood = likelihood
      
      for ex, (vSen, aSen) in enumerate(zip(self.vCorpus, self.aCorpus)):
        if ex in self.testIndices:
          continue
        forwardProbs = self.forward(vSen, aSen, debug=False)
        backwardProbs = self.backward(vSen, aSen, debug=False) 
        if debug:
          print('forward prob: ', forwardProbs)
          print('backward prob: ', backwardProbs)
        initCounts[len(vSen)] += self.updateInitialCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        transCounts[len(vSen)] += self.updateTransitionCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        stateCounts = self.updateStateCounts(forwardProbs, backwardProbs) 
        aBigram = aSen[:-1, :, np.newaxis] * aSen[1:, np.newaxis, :]
        for i_ph in self.audioFeatDim:
          bigramCounts[:, i_ph] += np.sum(stateCounts[1:], axis=1).T @ aBigram[:, i_ph]
        unigramCounts += np.sum(stateCounts[1:], axis=1).T @ aSen
        conceptCounts[ex] += self.updateConceptCounts(vSen, aSen)
        self.conceptCountsA[ex] += np.sum(stateCounts, axis=1)
      self.conceptCounts = conceptCounts

      # Normalize
      for m in self.lenProb:
        self.init[m] = initCounts[m] / np.sum(initCounts[m]) 
        if debug:
          print('self.init: ', self.init[m])

      for m in self.lenProb:
        totCounts = np.sum(transCounts[m], axis=1)
        for s in range(m):
          if totCounts[s] == 0:
            # Not updating the transition arc if it is not used          
            self.trans[m][s] = self.trans[m][s]
          else:
            self.trans[m][s] = transCounts[m][s] / totCounts[s]
        if debug:
          print('self.trans: ', self.trans[m])
      
      normFactor = np.sum(np.maximum(bigramCounts, EPS), axis=-1) 
      self.bigramObs = bigramCounts / normFactor[:, :, np.newaxis]
      normFactor = np.sum(np.maximum(unigramCounts, EPS), axis=-1) 
      self.unigramObs = unigramCounts / normFactor[:, np.newaxis]

      self.updateSoftmaxWeight(conceptCounts, debug=False) 

      if (epoch + 1) % 10 == 0:
        self.lr /= 10

      if printStatus:
        print('Epoch %d takes %.2f s to finish' % (epoch, time.time() - begin_time))

    np.save(self.modelName+'_likelihoods.npy', likelihoods) 
   
  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x Dx matrix storing the phone sequence (Dx = size of the phone set) 
  #
  # Outputs:
  # -------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  def forward(self, vSen, aSen, debug=False):
    T = len(aSen)
    nState = len(vSen)
    forwardProbs = np.zeros((T, nState, self.nWords))   
    #if debug:
    #  print('self.lenProb.keys: ', self.lenProb.keys())
    #  print('init keys: ', self.init.keys())
    #  print('nState: ', nState)
    
    probs_z_given_y = self.softmaxLayer(vSen) 
    
    forwardProbs[0] = np.tile(self.init[nState][:, np.newaxis], (1, self.nWords)) * probs_z_given_y * (self.unigramObs @ aSen[0])
    for t in range(T-1):
      prob_x_t_given_y = self.bigramObs[:, np.argmax(aSen[t])] @ aSen[t+1]
      probs_x_t_z_given_y = probs_z_given_y * prob_x_t_given_y
      trans_diag = np.diag(np.diag(self.trans[nState]))
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      # Compute the diagonal term
      forwardProbs[t+1] += (trans_diag @ forwardProbs[t]) * prob_x_t_given_y 
      # Compute the off-diagonal term 
      if debug:
        print('probs_x_t_given_y: ', probs_z_given_y * (self.bigramObs[:, np.argmax(aSen[t])] @ aSen[t+1]))
        print('probs_x_t_z_given_y: ', probs_x_t_z_given_y)
        print('diag term: ', (trans_diag @ forwardProbs[t]) * prob_x_t_given_y)
        print('off diag term: ', trans_off_diag.T @ np.sum(forwardProbs[t], axis=-1))

      forwardProbs[t+1] += ((trans_off_diag.T @ np.sum(forwardProbs[t], axis=-1)) * probs_x_t_z_given_y.T).T 
       
    return forwardProbs

  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x Dx matrix storing the phone sequence
  # 
  # Outputs:
  # -------
  #   backwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y) 
  def backward(self, vSen, aSen, debug=False):
    T = len(aSen)
    nState = len(vSen)
    backwardProbs = np.zeros((T, nState, self.nWords))
    probs_z_given_y = self.softmaxLayer(vSen)

    backwardProbs[T-1] = 1.
    for t in range(T-1, 0, -1):
      prob_x_t_z_given_y = probs_z_given_y * (self.bigramObs[:, np.argmax(aSen[t-1])] @ aSen[t]) 
      backwardProbs[t-1] += np.diag(np.diag(self.trans[nState])) @ (backwardProbs[t] * (self.bigramObs[:, np.argmax(aSen[t-1])] @ aSen[t])) 
      if debug:
        print('backwardProbs[t-1]: ', backwardProbs[t-1])
 
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      backwardProbs[t-1] += np.tile(trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.nWords))
      if debug:
        print('diag term: ', np.diag(np.diag(self.trans[nState])) @ (backwardProbs[t] * (self.bigramObs[:, np.argmax(aSen[t-1])] @ aSen[t])))
        print('beta term for off-diag: ', trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)) 
        print('backwardProbs[t-1]: ', backwardProbs[t-1])
        print('off-diag term: ', trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1))
 
    return backwardProbs  

  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x Dx matrix storing the phone sequence
  #
  # Outputs:
  # -------
  #   initExpCounts: Tx x Ty maxtrix storing p(i_{t-1}, i_t|x, y) 
  def updateInitialCounts(self, forwardProbs, backwardProbs, vSen, aSen, debug=False):
    #assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all() 
    nState = len(vSen)
    T = len(aSen)
    # Update the initial prob  
    initExpCounts = np.zeros((nState,))  
    for t in range(T):
      # XXX
      initExpCounts += np.sum(np.maximum(forwardProbs[t] * backwardProbs[t], EPS), axis=-1) / np.sum(np.maximum(forwardProbs[t] * backwardProbs[t], EPS))
      if debug:
        #print('forwardProbs, backwardProbs: ', forwardProbs[t], backwardProbs[t])    
        print('np.sum(forward*backward): ', np.sum(forwardProbs[t] * backwardProbs[t])) 
    if debug:
      print('initExpCounts: ', initExpCounts)
 
    return initExpCounts

  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #   vSen: Ty x Dx matrix storing the image feature (e.g., VGG16 hidden activations)
  #   aSen: Tx x Dy matrix storing the phone sequence
  #
  # Outputs:
  # -------
  #   transExpCounts: Tx x Ty maxtrix storing p(i_{t-1}, i_t|x, y)
  def updateTransitionCounts(self, forwardProbs, backwardProbs, vSen, aSen, debug=False):
    nState = len(vSen)
    T = len(aSen) 
    transExpCounts = np.zeros((nState, nState))

    # Update the transition probs
    probs_z_given_y = self.softmaxLayer(vSen)
    for t in range(T-1):
      prob_x_t_z_given_y = probs_z_given_y * (self.bigramObs[:, np.argmax(aSen[t])] @ aSen[t+1]) 
      prob_x_t_given_z = (self.bigramObs[:, np.argmax(aSen[t])] @ aSen[t+1])
      alpha = np.tile(np.sum(forwardProbs[t], axis=-1)[:, np.newaxis], (1, nState)) 
      trans_diag = np.tile(np.diag(self.trans[nState])[:, np.newaxis], (1, self.nWords))
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      transExpCount = np.zeros((nState, nState)) 
      transExpCount += np.diag(np.sum(forwardProbs[t] * trans_diag * prob_x_t_given_z * backwardProbs[t+1], axis=-1))
      transExpCount += alpha * trans_off_diag * np.sum(prob_x_t_z_given_y * backwardProbs[t+1], axis=-1)
      
      if debug:
        print('diag count: ', np.diag(np.sum(forwardProbs[t] * trans_diag * prob_x_t_given_z * backwardProbs[t+1], axis=-1)))
        print('diag count: ', alpha * trans_off_diag * np.sum(prob_x_t_z_given_y * backwardProbs[t+1], axis=-1))
        print("transExpCount: ", transExpCount)
      # XXX
      transExpCount = np.maximum(transExpCount, EPS) / np.sum(np.maximum(transExpCount, EPS))

      # Maintain Toeplitz assumption
      if len(self.lenProb) >= 6:
        # XXX
        transJumpCount = {}
        for s in range(nState):
          for next_s in range(nState):
            if next_s - s not in transJumpCount:
              #if DEBUG:
              #  print('new jump: ', next_s - s) 
              transJumpCount[next_s - s] = transExpCount[s][next_s]
            else:
              transJumpCount[next_s - s] += transExpCount[s][next_s]

        for s in range(nState):
          for next_s in range(nState):
            transExpCounts[s][next_s] += transJumpCount[next_s - s]
      else:
        transExpCounts += transExpCount
    return transExpCounts
   
  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #
  # Outputs:
  # -------
  #   newStateCounts: Tx x Ty x K maxtrix storing p(z_{i_t}|x, y) 
  def updateStateCounts(self, forwardProbs, backwardProbs):
    #assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all()
    T = forwardProbs.shape[0]
    nState = forwardProbs.shape[1]
    normFactor = np.maximum(np.sum(np.sum(forwardProbs * backwardProbs, axis=-1), axis=-1), EPS)
    newStateCounts = np.transpose(np.transpose(forwardProbs * backwardProbs, (1, 2, 0)) / normFactor, (2, 0, 1)) 
   
    return newStateCounts

  # Inputs:
  # ------
  #   vSen: Ty x Dx matrix storing the image feature (e.g., VGG16 hidden activations)
  #   aSen: Tx x Dy matrix storing the phone sequence
  #
  # Outputs:
  # -------
  #   newConceptCounts: Ty x K maxtrix storing p(z_i|x, y) 
  def updateConceptCounts(self, vSen, aSen, debug=False):
    T = len(aSen)
    nState = vSen.shape[0] 
    newConceptCounts = np.zeros((nState, self.nWords)) 
    probs_x_given_y_concat = np.zeros((T, nState * self.nWords, nState))
    probs_z_given_y = self.softmaxLayer(vSen)
    for i in range(nState):
      for k in range(self.nWords):
        probs_z_given_y_ik = deepcopy(probs_z_given_y)
        probs_z_given_y_ik[i] = 0.
        probs_z_given_y_ik[i, k] = 1.
        probs_x_given_y_concat[0, i*self.nWords+k, :] = (probs_z_given_y_ik @ (self.unigramObs @ aSen.T)).T
        # TODO
        probs_x_given_y_concat[1:, i*self.nWords+k, :] = (probs_z_given_y_ik @ np.sum(self.bigramObs[:, np.argmax(aSen[:-1], axis=1)] * aSen[1:], axis=-1)).T

    forwardProbsConcat = np.zeros((nState * self.nWords, nState))
    forwardProbsConcat = self.init[nState] * probs_x_given_y_concat[0]
    for t in range(T-1):
      forwardProbsConcat = (forwardProbsConcat @ self.trans[nState]) * probs_x_given_y_concat[t+1]

    newConceptCounts = np.sum(forwardProbsConcat, axis=-1).reshape((nState, self.nWords))
    newConceptCounts = ((probs_z_given_y * newConceptCounts).T / np.sum(probs_z_given_y * newConceptCounts, axis=1)).T 
    if debug:
      print(newConceptCounts)
    return newConceptCounts

  # Inputs:
  # ------
  #   conceptCounts: a list of Ty x K matrices storing p(z_i|x, y) for each utterances
  #   numGDIterations: int, number of gradient descent iterations   
  #
  # Outputs:
  # -------
  #   None
  def updateSoftmaxWeight(self, conceptCounts, debug=False):
    dW = np.zeros((self.nWords, self.imageFeatDim + 1))
    
    for vSen, conceptCount in zip(self.vCorpus, conceptCounts):
      N = vSen.shape[0]
      # XXX
      vConcat = np.concatenate([vSen, np.ones((N, 1))], axis=1)
      zProb = self.softmaxLayer(vSen, debug=debug) 
      dW += 1. / len(self.vCorpus) * (conceptCount - zProb).T @ vConcat 
      if debug:
        print('dW: ', dW)
      
    self.W[:, :-1] = (1. - self.momentum) * self.W[:, :-1] + self.lr * dW[:, :-1]
    self.W[:, -1] =  (1. - self.momentum) * self.W[:, -1] + self.lr * dW[:, -1]

  # Compute translation length probabilities q(m|n)
  def computeTranslationLengthProbabilities(self, smoothing=None):
      # Implement this method
      #pass        
      #if DEBUG:
      #  print(len(self.tCorpus))
      for ts, fs in zip(self.vCorpus, self.aCorpus):
        # len of ts contains the NULL symbol
        #if len(ts)-1 not in self.lenProb.keys():
        self.lenProb[len(ts)] = {}
        if len(fs) not in self.lenProb[len(ts)].keys():
          self.lenProb[len(ts)][len(fs)] = 1
        else:
          self.lenProb[len(ts)][len(fs)] += 1
      
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
       
      for tl in self.lenProb.keys():
        totCount = sum(self.lenProb[tl].values())  
        for fl in self.lenProb[tl].keys():
          self.lenProb[tl][fl] = self.lenProb[tl][fl] / totCount 

  def computeAvgLogLikelihood(self):
    ll = 0.
    for vSen, aSen in zip(self.vCorpus, self.aCorpus):
      forwardProb = self.forward(vSen, aSen)
      #backwardProb = self.backward(tSen, fSen)
      # XXX
      likelihood = np.maximum(np.sum(forwardProb[-1]), EPS)
      ll += math.log(likelihood)
    return ll / len(self.vCorpus)
  
  def softmaxLayer(self, vSen, debug=False):
    N = vSen.shape[0]
    vConcat = np.concatenate([vSen, np.ones((N, 1))], axis=1)
    prob = vConcat @ self.W.T
    #prob = vSen @ self.W.T 
    if debug:
      print('prob: ', prob)
    prob = np.exp(prob.T - logsumexp(prob, axis=1)).T
    return prob

  def align(self, aSen, vSen, unkProb=10e-12, debug=False):
    nState = len(vSen)
    T = len(aSen)
    scores = np.zeros((nState,))
    probs_z_given_y = self.softmaxLayer(vSen)
    
    backPointers = np.zeros((T, nState), dtype=int)
    # TODO
    probs_x_given_y = np.zeros((T, nState))
    probs_x_given_y[0] = probs_z_given_y @ (self.unigramObs @ aSen[0])
    probs_x_given_y[1:] = (probs_z_given_y @ (self.np.sum(bigramObs[:, np.argmax(aSen[:-1], axis=-1)] * aSen[1:], axis=-1))).T 
    if debug:
      print('probs_z_given_y: ', probs_z_given_y)
      print('probs_x_given_y: ', probs_x_given_y)
    
    scores = self.init[nState] * probs_x_given_y[0]
    if debug:
      print('scores: ', scores)

    alignProbs = [scores.tolist()] 
    for t in range(1, T):
      candidates = np.tile(scores, (nState, 1)).T * self.trans[nState] * probs_x_given_y[t]
      backPointers[t] = np.argmax(candidates, axis=0)
      # XXX
      scores = np.maximum(np.max(candidates, axis=0), EPS)
      if debug:
        print('self.init: ', self.init[nState])
        print('self.trans: ', self.trans[nState])
        print('backPtrs: ', backPointers[t])
        print('candidates: ', candidates)

      alignProbs.append((scores / np.sum(scores)).tolist())
      
      #if DEBUG:
      #  print(scores)
    
    curState = np.argmax(scores)
    bestPath = [int(curState)]
    for t in range(T-1, 0, -1):
      if DEBUG:
        print('curState: ', curState)
      curState = backPointers[t, curState]
      bestPath.append(int(curState))
    
    return bestPath[::-1], alignProbs

  def cluster(self, aSen, vSen, alignment):
    nState = len(vSen)
    T = len(aSen)
    probs_z_given_y = self.softmaxLayer(vSen)
    probs_x_given_z = np.zeros((T, self.nWords))
    probs_x_given_z[0] = aSen[0] @ self.unigramObs.T
    probs_x_given_z[1:] = np.sum(aSen[1:] * self.bigramObs[:, np.argmax(aSen[:-1], axis=-1)], axis=-1).T
    scores = np.zeros((nState, self.nWords))
    scores += probs_z_given_y
    for i in range(nState):
      for t in range(T):
        if alignment[t] == i:
          scores[i] *= probs_x_given_z[t]
    return np.argmax(scores, axis=1).tolist(), scores.tolist()
    
  def printModel(self, fileName):
    initFile = open(fileName+'_initialprobs.txt', 'w')
    for nState in sorted(self.lenProb):
      for i in range(nState):
        initFile.write('%d\t%d\t%f\n' % (nState, i, self.init[nState][i]))
    initFile.close()

    transFile = open(fileName+'_transitionprobs.txt', 'w')
    for nState in sorted(self.lenProb):
      for i in range(nState):
        for j in range(nState):
          transFile.write('%d\t%d\t%d\t%f\n' % (nState, i, j, self.trans[nState][i][j]))
    transFile.close()

    np.save(fileName+'_unigram_observationprobs.npy', self.unigramObs)
    np.save(fileName+'_bigram_observationprobs.npy', self.bigramObs)
    with open(fileName+'_phone2idx.json', 'w') as f:
      json.dump(self.phone2idx, f)
    

  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, isPhoneme=True, debug=False):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    #if DEBUG:
    #  print(len(self.aCorpus))
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      alignment, alignProbs = self.align(aSen, vSen, debug=debug)
      clusters, clusterProbs = self.cluster(aSen, vSen, alignment)
      conceptAlignment = np.argmax(self.conceptCountsA[i], axis=1).tolist()
      if DEBUG:
        print(aSen, vSen)
        print(type(alignment[1]))
      align_info = {
            'index': i,
            'image_concepts': clusters,
            'concept_alignment': conceptAlignment,
            'alignment': alignment,
            'align_probs': alignProbs,
            'is_phoneme': isPhoneme
          }
      aligns.append(align_info)
      for a in alignment:
        f.write('%d ' % a)
      f.write('\n\n')
    f.close()
    
    # Write to a .json file for evaluation
    with open(filePrefix+'.json', 'w') as f:
      json.dump(aligns, f, indent=4, sort_keys=True)            

if __name__ == '__main__':
  tasks = [2]
  #----------------------------#
  # Word discovery on tiny.txt #
  #----------------------------#
  if 0 in tasks:
    speechFeatureFile = 'tiny.txt'
    imageFeatureFile = 'tiny.npz'   
    image_feats = {'arr_0':np.array([[1., 0., 0.], [0., 1., 0.]]), 'arr_1':np.array([[0., 1., 0.], [0., 0., 1.]]), 'arr_2':np.array([[0., 0., 1.], [1., 0., 0.]])}   
    audio_feats = '0 1\n1 2\n2 0'
    with open('tiny.txt', 'w') as f:
      f.write(audio_feats)
    np.savez('tiny.npz', **image_feats)
    modelConfigs = {'has_null': False, 'n_words': 3, 'momentum': 0., 'learning_rate': 0.01}
    model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/jan_14_tiny/tiny')
    model.trainUsingEM(30, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=100, T0=1., debug=False) 
    model.printAlignment('exp/jan_14_tiny/tiny', debug=False)
  #-------------------------------#
  # Feature extraction for MSCOCO #
  #-------------------------------#
  if 1 in tasks:
    featType = 'gaussian'    
    speechFeatureFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt'
    imageConceptFile = '../data/mscoco/trg_mscoco_subset_subword_level_power_law.txt'
    imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    conceptIdxFile = 'exp/dec_30_mscoco/concept2idx.json'

    vCorpus = {}
    concept2idx = {}
    nTypes = 0
    with open(imageConceptFile, 'r') as f:
      vCorpusStr = []
      for line in f:
        vSen = line.strip().split()
        vCorpusStr.append(vSen)
        for vWord in vSen:
          if vWord not in concept2idx:
            concept2idx[vWord] = nTypes
            nTypes += 1
    
    # Generate nTypes different clusters
    imgFeatDim = 2
    centroids = 10 * np.random.normal(size=(nTypes, imgFeatDim)) 
     
    for ex, vSenStr in enumerate(vCorpusStr):
      N = len(vSenStr)
      if featType == 'one-hot':
        vSen = np.zeros((N, nTypes))
        for i, vWord in enumerate(vSenStr):
          vSen[i, concept2idx[vWord]] = 1.
      elif featType == 'gaussian':
        vSen = np.zeros((N, imgFeatDim))
        for i, vWord in enumerate(vSenStr):
          vSen[i] = centroids[concept2idx[vWord]] + 0.1 * np.random.normal(size=(imgFeatDim,))
      vCorpus['arr_'+str(ex)] = vSen

    np.savez(imageFeatureFile, **vCorpus)
    with open(conceptIdxFile, 'w') as f:
      json.dump(concept2idx, f, indent=4, sort_keys=True)
  #--------------------------#
  # Word discovery on MSCOCO #
  #--------------------------#
  if 2 in tasks:      
    speechFeatureFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt'
    #imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    #imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_vectors.npz'
    imageFeatureFile = '../data/mscoco/mscoco_vgg_penult.npz'
    modelConfigs = {'has_null': False, 'n_words': 65, 'momentum': 0.0, 'learning_rate': 0.01, 'normalize_vfeat': False, 'step_scale': 5}
    modelName = 'exp/jan_18_mscoco_vgg16_momentum%.1f_lr%.1f_stepscale%d_linear/image_phone' % (modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale']) 
    print(modelName)
    #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
    model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=300, T0=50., debug=False)
    #model.printAlignment('exp/dec_30_mscoco/image_phone_alignment', debug=False) 
    model.printAlignment(modelName+'_alignment', debug=False)
  #-----------------------------#
  # Word discovery on Flickr30k #
  #-----------------------------#
  if 3 in tasks:
    speechFeatureFile = '../data/flickr30k/phoneme_level/flickr30k_no_NULL_top_100.txt'
    #imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    #imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_vectors.npz'
    imageFeatureFile = '../data/flickr30k/phoneme_level/flickr30k_no_NULL_top_100_vgg_penult.npz'
    modelConfigs = {'has_null': False, 'n_words': 100, 'momentum': 0.0, 'learning_rate': 0.01, 'normalize_vfeat': False, 'step_scale': 1.}
    modelName = 'exp/jan_21_flickr_vgg16_top100_momentum%.2f_lr%.5f_stepscale%d_linear/image_phone' % (modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale']) 
    print(modelName)
    #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
    model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=30, T0=50., debug=False)
    #model.printAlignment('exp/dec_30_mscoco/image_phone_alignment', debug=False) 
    model.printAlignment(modelName+'_alignment', debug=False) 
