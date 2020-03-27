import numpy as np
import math
import json
import time
from scipy.special import logsumexp
import random
from copy import deepcopy
from sklearn.cluster import KMeans

NULL = "NULL"
DEBUG = False
EPS = 1e-50
random.seed(1)
np.random.seed(1)

# A word discovery model using image regions and phones
# * The transition matrix is assumed to be Toeplitz 
class ImagePhoneGaussianHMMWordDiscoverer:
  def __init__(self, speechFeatureFile, imageFeatureFile, modelConfigs, modelName='image_phone_hmm_word_discoverer'):
    self.modelName = modelName 
    # Initialize data structures for storing training data
    self.aCorpus = []                   # aCorpus is a list of acoustic features

    self.vCorpus = []                   # vCorpus is a list of image posterior features (e.g. VGG softmax)
    self.hasNull = modelConfigs.get('has_null', False)
    self.nWords = modelConfigs.get('n_words', 66)
    self.width = modelConfigs.get('width', 1.) 
    self.momentum = modelConfigs.get('momentum', 0.)
    self.lr = modelConfigs.get('learning_rate', 10.)
    self.isExact = modelConfigs.get('is_exact', False)
    self.normalize_vfeat = modelConfigs.get('normalize_vfeat', False) 
    self.init = {}
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.obs = None                 # obs[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
    self.avgLogTransProb = float('-inf')
     
    # Read the corpus
    self.readCorpus(speechFeatureFile, imageFeatureFile, debug=False);
    self.initProbFile = modelConfigs.get('init_prob_file', None)
    self.transProbFile = modelConfigs.get('trans_prob_file', None)
    self.obsProbFile = modelConfigs.get('obs_prob_file', None)
    self.visualAnchorFile = modelConfigs.get('visual_anchor_file', None)
     
     
  def readCorpus(self, speechFeatFile, imageFeatFile, debug=False):
    aCorpus = []
    vCorpus = []
    self.phone2idx = {}
    nTypes = 0
    nPhones = 0
    nImages = 0

    vNpz = np.load(imageFeatFile)
    vCorpus = [vNpz[k] for k in sorted(vNpz.keys(), key=lambda x:int(x.split('_')[-1]))]
    if self.normalize_vfeat:
      vCorpus = [(vSen.T / np.linalg.norm(vSen, ord=2, axis=-1)).T for vSen in vCorpus]  
      if debug:
        print(np.linalg.norm(vCorpus[0], ord=2, axis=-1))
    if debug:
      print(len(vCorpus))
      print(vCorpus[0].shape)
    
    # XXX
    self.vCorpus = [vNpz[k] for k in sorted(vNpz, key=lambda x:int(x.split('_')[-1]))]
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
        if int(m) not in self.init:
          self.init[int(m)] = np.zeros((int(m),))
        self.init[int(m)][int(s)] = float(prob)
      f.close()
    
    if self.transProbFile:
      f = open(self.transProbFile)
      for line in f:
        m, cur_s, next_s, prob = line.split()
        if int(m) not in self.trans:
          self.trans[int(m)] = np.zeros((int(m), int(m)))
        self.trans[int(m)][int(cur_s)][int(next_s)] = float(prob)     
      f.close()
    
    if self.obsProbFile:
      self.obs = np.load(self.obsProbFile)
    else:
      self.obs = 1. / self.audioFeatDim * np.ones((self.nWords, self.audioFeatDim))

    # XXX    
    if self.visualAnchorFile:
      self.mus = np.load(self.visualAnchorFile)
    else:
      #self.mus = 10. * np.eye(self.nWords)
      self.mus = KMeans(n_clusters=self.nWords).fit(np.concatenate(self.vCorpus, axis=0)).cluster_centers_
      #self.mus = 1. * np.random.normal(size=(self.nWords, self.imageFeatDim))
    print("Finish initialization after %0.3f s" % (time.time() - begin_time))
    self.printUnimodalCluster(filePrefix=self.modelName)
  
  # TODO
  # Inputs:
  # ------
  #   T0: positive value, initial temperature
  #       numIterations: number of SA iterations
  #   stepSize: scale of the random jump step
  #
  # Outputs:
  # -------
  #   None
  def simulatedAnnealing(self, numIterations=100, T0=0.5, stepScale=5., debug=False):
    # Initial negative log-likelihood as the initial energy 
    self.trainUsingEM(numIterations=5, warmStart=False, printStatus=True)
    E0 = -self.computeAvgLogLikelihood() 
    Emin = E0 
    count = 0
    for epoch in range(numIterations):
      print('Simulated Annealing Iteration %d' % epoch)
      begin_time = time.time()
      init_prev = deepcopy(self.init)
      trans_prev = deepcopy(self.trans)
      obs_prev = deepcopy(self.obs) 
      mus_prev = deepcopy(self.mus)
      self.mus += stepScale * np.random.normal(size=(self.nWords, self.imageFeatDim))
      self.trainUsingEM(numIterations=5, warmStart=True, printStatus=False)
      E1 = -self.computeAvgLogLikelihood() 
      print('Current and previous energy level: ', E1, E0)
      # Cooling scheme
      Tk = T0 / np.log(epoch+2)
      # Random jump according to the Boltzman distribution with dE = E1 - E0
      # 1) Continue jumping if not finding a good local minimum and choosing not to stay at the bad local optimum 
      if E1 > E0 and random.random() > np.exp(-(E1 - E0) / Tk): 
        self.mus = mus_prev
        self.init = init_prev
        self.trans = trans_prev
        self.obs = obs_prev
      # 2) Otherwise, transition to the new energy level; save the weight if it is the lowest energy level so far 
      else:
        if debug:
          print('Random jump at temperature %.5f' % Tk)
        E0 = E1
        if E1 < Emin:
          Emin = E1
          count += 1
          print('Update %d after %.2f s: current lowest energy level is %.5f' % (count, time.time()-begin_time, Emin))
          self.printModel(self.modelName+'_%d' % count)
          self.printAlignment(self.modelName+'_%d_alignment' % count, debug=False)
          begin_time = time.time()

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
      phoneCounts = np.zeros((self.nWords, self.audioFeatDim))      
      conceptCounts = [np.zeros((vSen.shape[0], self.nWords)) for vSen in self.vCorpus]
      self.conceptCounts = conceptCounts
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
        forwardProbs = self.forward(vSen, aSen, debug=False)
        backwardProbs = self.backward(vSen, aSen, debug=False) 
        if debug:
          print('forward prob: ', forwardProbs)
          print('backward prob: ', backwardProbs)
        initCounts[len(vSen)] += self.updateInitialCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        transCounts[len(vSen)] += self.updateTransitionCounts(forwardProbs, backwardProbs, vSen, aSen, debug=False)
        stateCounts = self.updateStateCounts(forwardProbs, backwardProbs)
        phoneCounts += np.sum(stateCounts, axis=1).T @ aSen
        
        conceptCounts[ex] += self.updateConceptCounts(vSen, aSen)
        self.conceptCountsA[ex] += np.sum(stateCounts, axis=1)
      self.conceptCounts = conceptCounts

      # Normalize
      for m in self.lenProb:
        self.init[m] = np.maximum(initCounts[m], EPS) / np.sum(np.maximum(initCounts[m], EPS)) 

      for m in self.lenProb:
        totCounts = np.sum(np.maximum(transCounts[m], EPS), axis=1)
        for s in range(m):
          if totCounts[s] == 0:
            # Not updating the transition arc if it is not used          
            self.trans[m][s] = self.trans[m][s]
          else:
            self.trans[m][s] = np.maximum(transCounts[m][s], EPS) / totCounts[s]
      
      normFactor = np.sum(np.maximum(phoneCounts, EPS), axis=-1) 
      self.obs = (phoneCounts.T / normFactor).T
      
      if debug:
        print('phoneCounts: ', phoneCounts)
        print('self.obs: ', self.obs)
      
      # XXX
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
  #   (optional) restrictState: a tuple (i, k), where i is the alignment index and k is the image concept class  
  #
  # Outputs:
  # -------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  def forward(self, vSen, aSen, restrictState=None, debug=False):
    T = len(aSen)
    nState = len(vSen)
    forwardProbs = np.zeros((T, nState, self.nWords))   
    #if debug:
    #  print('self.lenProb.keys: ', self.lenProb.keys())
    #  print('init keys: ', self.init.keys())
    #  print('nState: ', nState)
    
    probs_z_given_y = self.softmaxLayer(vSen) 
    if restrictState is not None:
      probs_z_given_y[restrictState[0]] = 0.
      probs_z_given_y[restrictState[0], restrictState[1]] = 1.
    
    forwardProbs[0] = np.tile(self.init[nState][:, np.newaxis], (1, self.nWords)) * probs_z_given_y * (self.obs @ aSen[0])
    for t in range(T-1):
      prob_x_t_given_y = self.obs @ aSen[t+1]
      probs_x_t_z_given_y = probs_z_given_y * prob_x_t_given_y
      trans_diag = np.diag(np.diag(self.trans[nState]))
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      # Compute the diagonal term
      forwardProbs[t+1] += (trans_diag @ forwardProbs[t]) * prob_x_t_given_y 
      # Compute the off-diagonal term 
      if debug:
        print('probs_x_t_given_y: ', probs_z_given_y * (self.obs @ aSen[t]))
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
      prob_x_t_z_given_y = probs_z_given_y * (self.obs @ aSen[t]) 
      backwardProbs[t-1] += np.diag(np.diag(self.trans[nState])) @ (backwardProbs[t] * (self.obs @ aSen[t])) 
      if debug:
        print('backwardProbs[t-1]: ', backwardProbs[t-1])
 
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      backwardProbs[t-1] += np.tile(trans_off_diag @ np.sum(backwardProbs[t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.nWords))
      if debug:
        print('diag term: ', np.diag(np.diag(self.trans[nState])) @ (backwardProbs[t] * (self.obs @ aSen[t])))
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
      prob_x_t_z_given_y = probs_z_given_y * (self.obs @ aSen[t+1]) 
      prob_x_t_given_z = (self.obs @ aSen[t+1])
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

      # XXX
      # Reduce the number of parameters if the length of image-caption pairs vary too much by maintaining the Toeplitz assumption
      if len(self.lenProb) >= 6:
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
        probs_x_given_y_concat[:, i*self.nWords+k, :] = (probs_z_given_y_ik @ (self.obs @ aSen.T)).T

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
    if self.isExact:
      musNext = np.zeros((self.nWords, self.imageFeatDim))
      Delta = conceptCount - zProb 
      musNext += Delta.T @ vSen
      normFactor += np.sum(Delta, axis=0)
      normFactor = np.sign(normFactor) * np.maximum(np.abs(normFactor), EPS)  
      self.mus = (musNext.T / normFactor).T
    else:
      dmus = np.zeros((self.nWords, self.imageFeatDim))
      normFactor = np.zeros((self.nWords,))
      for vSen, conceptCount in zip(self.vCorpus, conceptCounts):
        zProb = self.softmaxLayer(vSen, debug=debug)
        Delta = conceptCount - zProb 
        dmus += 1. / (len(self.vCorpus) * self.width) * (Delta.T @ vSen - (np.sum(Delta, axis=0) * self.mus.T).T) 
      # XXX
      if debug:
        print('conceptCount: ', conceptCount)
        print('zProb: ', zProb)
        print('dmus: ', dmus)
      self.mus = (1. - self.momentum) * self.mus + self.lr * dmus

  def softmaxLayer(self, vSen, debug=False):
    N = vSen.shape[0]
    prob = np.zeros((N, self.nWords))
    for i in range(N):
      prob[i] = -np.sum((vSen[i] - self.mus) ** 2, axis=1) / self.width
    if debug:
      print('self.mus: ', self.mus)
      print('prob: ', prob)
    prob = np.exp(prob.T - logsumexp(prob, axis=1)).T
    return prob

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

  def align(self, aSen, vSen, unkProb=10e-12, debug=False):
    nState = len(vSen)
    T = len(aSen)
    scores = np.zeros((nState,))
    probs_z_given_y = self.softmaxLayer(vSen)
    
    backPointers = np.zeros((T, nState), dtype=int)
    probs_x_given_y = (probs_z_given_y @ (self.obs @ aSen.T)).T    
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

      alignProbs.append((scores / np.sum(np.maximum(scores, EPS))).tolist())      
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
    probs_x_given_z = aSen @ self.obs.T
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

    np.save(fileName+'_observationprobs.npy', self.obs)
   
    with open(fileName+'_phone2idx.json', 'w') as f:
      json.dump(self.phone2idx, f)
   
    np.save(fileName+'_visualanchors.npy', self.mus) 

  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, isPhoneme=True, debug=False):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    #if DEBUG:
    #  print(len(self.aCorpus))
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      alignment, alignProbs = self.align(aSen, vSen, debug=debug)
      clustersV, clusterProbs = self.cluster(aSen, vSen, alignment)
      conceptAlignment = np.argmax(self.conceptCounts[i], axis=1).tolist() 

      if DEBUG:
        print(aSen, vSen)
        print(type(alignment[1]))
      align_info = {
            'index': i,
            'image_concepts': clustersV,
            'concept_alignment': conceptAlignment,
            'alignment': alignment,
            'align_probs': alignProbs,
            'concept_probs': self.conceptCounts[i].tolist(),
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
  def printUnimodalCluster(self, filePrefix):
    f = open(filePrefix+'.txt', 'w')
    cluster_infos = []
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      clusterProbs = self.softmaxLayer(vSen)
      clusters = np.argmax(clusterProbs, axis=1)
      
      cluster_info = {
          'index': i,
          'image_concepts': clusters.tolist(),
          'cluster_probs': clusterProbs.tolist()
        }
      cluster_infos.append(cluster_info)
    
    with open(filePrefix+'.json', 'w') as f:
      json.dump(cluster_infos, f, indent=4, sort_keys=True)

if __name__ == '__main__':
  tasks = [0]
  #----------------------------#
  # Word discovery on tiny.txt #
  #----------------------------#
  if 0 in tasks:
    speechFeatureFile = 'tiny.txt'
    imageFeatureFile = 'tiny.npz'   
    image_feats = {'arr_0':np.array([[1., 0., 0.], [0., 1., 0.]]), 'arr_1':np.array([[0., 1., 0.], [0., 0., 1.]]), 'arr_2':np.array([[0., 0., 1.], [1., 0., 0.]])}   
    audio_feats = '0 1\n1 2\n2 0'
    exp_dir = 'exp/jan_14_tiny/'
    with open(exp_dir + 'tiny.txt', 'w') as f:
      f.write(audio_feats)
    np.savez(exp_dir + 'tiny.npz', **image_feats)
    modelConfigs = {'has_null': False, 'n_words': 3, 'momentum': 0., 'learning_rate': 1.}
    model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/jan_14_tiny/tiny')
    model.trainUsingEM(30, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=100, T0=50., debug=False) 
    model.printAlignment(exp_dir+'tiny', debug=False)
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
    modelConfigs = {'has_null': False, 'n_words': 65, 'momentum': 0.0, 'learning_rate': 0.1, 'normalize_vfeat': False, 'step_scale': 5}
    modelName = 'exp/jan_18_mscoco_vgg16_momentum%.2f_lr%.5f_stepscale%d_gaussiansoftmax/image_phone' % (modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale']) 
    print(modelName)
    #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
    model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=30, T0=50., debug=False)
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
    modelConfigs = {'has_null': False, 'n_words': 100, 'momentum': 0.0, 'learning_rate': 0.1, 'normalize_vfeat': False, 'step_scale': 1.}
    modelName = 'exp/jan_20_flickr_vgg16_top100_momentum%.2f_lr%.5f_stepscale%d_gaussiansoftmax/image_phone' % (modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale']) 
    print(modelName)
    #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
    model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(10, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=30, T0=50., debug=False)
    #model.printAlignment('exp/dec_30_mscoco/image_phone_alignment', debug=False) 
    model.printAlignment(modelName+'_alignment', debug=False)
  if 4 in tasks:
    speechFeatureFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt' 
    imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    conceptIdxFile = 'exp/dec_30_mscoco/concept2idx.json'
    exp_dir = 'exp/mscoco2k_gaussian_synthetic_momentum0.0_lr0.10000_stepscale0.10/'
    modelConfigs = {'has_null': False, 'n_words': 65, 'momentum': 0.0, 'learning_rate': 0.1, 'normalize_vfeat': False, 'step_scale': 1.}
    '''                'init_prob_file': exp_dir + 'image_phone_40_iter=24.txt_initialprobs.txt', 
                    'trans_prob_file': exp_dir + 'image_phone_40_iter=24.txt_transitionprobs.txt', 
                    'obs_prob_file': exp_dir + 'image_phone_40_iter=24.txt_observationprobs.npy', 
                    'visual_anchor_file': exp_dir + 'image_phone_40_iter=24.txt_visualanchors.npy'}'''
    modelName = 'exp/feb_4_test_posterior_and_decoding/image_phone'
    model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(10, writeModel=False)
  if 5 in tasks:
    modelConfigs = {'has_null': False, 'n_words': 65, 'momentum': 0.0, 'learning_rate': 0.1, 'normalize_vfeat': False, 'step_scale': 1.}
    modelName = 'exp/feb_5_test_cluster_quality/image_phone_unimodal_clusters'
    speechFeatureFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt' 
    imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    model = ImagePhoneGaussianHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(2, writeModel=True)
    model.printUnimodalCluster(filePrefix=modelName)
