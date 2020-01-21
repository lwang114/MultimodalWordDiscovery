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
# TODO
class ImagePhoneHMMDNNWordDiscoverer:
  def __init__(self, speechFeatureFile, imageFeatureFile, modelConfigs, initProbFile=None, transProbFile=None, obsProbFile=None, modelName='image_phone_hmm_word_discoverer'):
    self.modelName = modelName 
    # Initialize data structures for storing training data
    self.aCorpus = []                   # aCorpus is a list of acoustic features
    self.vCorpus = []                   # vCorpus is a list of image posterior features (e.g. VGG softmax)
    self.hasNull = modelConfigs.get('has_null', False)
    self.nWords = modelConfigs.get('n_words', 66) 
    self.hiddenDim = modelConfigs.get('hidden_dim', 100)
    self.momentum = modelConfigs.get('momentum', 0.)
    self.lr = modelConfigs.get('learning_rate', 10.) 
    self.normalize_vfeat = modelConfigs.get('normalize_vfeat', False) 
    self.init = {}
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.obs = None                 # obs[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
    self.avgLogTransProb = float('-inf')
     
    # Read the corpus
    self.readCorpus(speechFeatureFile, imageFeatureFile, debug=False);
    self.initProbFile = initProbFile
    self.transProbFile = transProbFile
    self.obsProbFile = obsProbFile
     
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

    if self.obsProbFile:
      self.obs = np.load(self.obsProbFile)
    else:
      self.obs = 1. / self.audioFeatDim * np.ones((self.nWords, self.audioFeatDim))
    
    # XXX
    #self.W = 10.*np.eye(self.hiddenDim)[:self.nWords]
    self.V = 1. * np.random.normal(size=(self.hiddenDim, self.imageFeatDim)) 
    self.W = 1. * np.random.normal(size=(self.nWords, self.hiddenDim))  
    print("Finish initialization after %0.3f s" % (time.time() - begin_time))
  
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
    Tk = T0
    for epoch in range(numIterations):
      print('Simulated Annealing Iteration %d' % epoch)
      begin_time = time.time()
      init_prev = deepcopy(self.init)
      trans_prev = deepcopy(self.trans)
      obs_prev = deepcopy(self.obs) 
      W_prev = deepcopy(self.W)
      V_prev = deepcopy(self.V)
      self.W += stepScale * np.random.normal(size=(self.nWords, self.hiddenDim))
      self.V += 0.1 * stepScale * np.random.normal(size=(self.hiddenDim, self.imageFeatDim))
      self.trainUsingEM(numIterations=10, warmStart=True, printStatus=False)
      E1 = -self.computeAvgLogLikelihood() 
      print('Current and previous energy level: ', E1, E0)
      # Cooling scheme
      Tk = T0 / np.log(epoch+2)
      # Random jump according to the Boltzman distribution with dE = E1 - E0
      # 1) Continue jumping if not finding a good local minimum and choosing not to stay at the bad local optimum 
      if E1 > E0 and random.random() > np.exp(-(E1 - E0) / Tk): 
        if debug:
          print('Stay at energy level %.5f with probability %.5f' % (E0, 1 - np.exp(-(E1 - E0) / Tk)))
        self.W = W_prev
        self.V = V_prev
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
    for epoch in range(numIterations): 
      begin_time = time.time()
      initCounts = {m: np.zeros((m,)) for m in self.lenProb}
      transCounts = {m: np.zeros((m, m)) for m in self.lenProb}
      phoneCounts = np.zeros((self.nWords, self.audioFeatDim))      
      conceptCounts = [np.zeros((vSen.shape[0], self.nWords)) for vSen in self.vCorpus]
    
      if printStatus:
        likelihood = self.computeAvgLogLikelihood()
        print('Epoch', epoch, 'Average Log Likelihood:', likelihood)
        if writeModel and likelihood > maxLikelihood:
          self.printModel(self.modelName + '_iter='+str(epoch)+'.txt')
          model.printAlignment(self.modelName+'_iter='+str(epoch)+'_alignment', debug=False)          
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
        # XXX switch to approximate p(z_i|x, y) if the number of concepts
        # is too large for exact computation
        if self.nWords > 50:
          conceptCounts[ex] += np.mean(stateCounts, axis=0)
        else:
          conceptCounts[ex] += self.updateConceptCounts(vSen, aSen)

      # Normalize
      # XXX
      for m in self.lenProb:
        self.init[m] = np.maximum(initCounts[m], EPS) / np.sum(np.maximum(initCounts[m], EPS)) 
        if debug:
          print('self.init: ', self.init[m])

      # XXX
      for m in self.lenProb:
        totCounts = np.sum(np.maximum(transCounts[m], EPS), axis=1)
        for s in range(m):
          if totCounts[s] == 0:
            # Not updating the transition arc if it is not used          
            self.trans[m][s] = self.trans[m][s]
          else:
            self.trans[m][s] = np.maximum(transCounts[m][s], EPS) / totCounts[s]
        if debug:
          print('self.trans: ', self.trans[m])
      
      # XXX 
      normFactor = np.sum(np.maximum(phoneCounts, EPS), axis=-1) 
      self.obs = (phoneCounts.T / normFactor).T
      
      if debug:
        print('phoneCounts: ', phoneCounts)
        print('self.obs: ', self.obs)
      # XXX
      self.updateNeuralNetWeights(conceptCounts, debug=False) 

      if (epoch + 1) % 10 == 0:
        self.lr /= 10
        
      if printStatus:
        print('Epoch %d takes %.2f s to finish' % (epoch, time.time() - begin_time))

  # Inputs:
  # ------
  #   vSen: Ty x Dy matrix storing the image feature (e.g., VGG 16 hidden activation)
  #   aSen: Tx x Dx matrix storing the phone sequence (Dx = size of the phone set) 
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
    
    probs_z_given_y = self.softmaxLayer(self.hiddenLayer(vSen)) 
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
    probs_z_given_y = self.softmaxLayer(self.hiddenLayer(vSen))

    backwardProbs[T-1] = deepcopy(probs_z_given_y)
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
    probs_z_given_y = self.softmaxLayer(self.hiddenLayer(vSen))
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
  def updateConceptCounts(self, vSen, aSen):
    nState = vSen.shape[0]
    newConceptCounts = np.zeros((nState, self.nWords)) 
    
    for i in range(nState):
      for k in range(self.nWords):
        forwardProbs = self.forward(vSen, aSen, restrictState=[i, k])
        newConceptCounts[i, k] = np.sum(forwardProbs[-1]) 

    newConceptCounts = (newConceptCounts.T / np.sum(newConceptCounts, axis=1)).T 
    return newConceptCounts

  # Inputs:
  # ------
  #   conceptCounts: a list of Ty x K matrices storing p(z_i|x, y) for each utterances
  #   numGDIterations: int, number of gradient descent iterations   
  #
  # Outputs:
  # -------
  #   None
  def updateNeuralNetWeights(self, conceptCounts, debug=False):
    dW = np.zeros((self.nWords, self.hiddenDim))
    dV = np.zeros((self.hiddenDim, self.imageFeatDim)) 
    for vSen, conceptCount in zip(self.vCorpus, conceptCounts):
      vHidden = self.hiddenLayer(vSen, debug=debug) 
      vProb = self.softmaxLayer(vHidden, debug=debug) 
      Delta = conceptCount - vProb
      Epsilon = Delta @ self.W
      
      if debug:
        print('Epsilon.shape: ', Epsilon.shape) 
      dW += Delta.T @ vHidden 
      dV += (Epsilon * vHidden * (1 - vHidden)).T @ vSen
      # Gradient for ReLU: dV += self.lr * (Epsilon * (vHidden > 0)).T @ vSen   
    self.W = (1. - self.momentum) * self.W + self.lr * dW
    self.V = (1. - self.momentum) * self.V + self.lr / 10. * dV  

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
  
  def hiddenLayer(self, vSen, debug=False, bias=False):
    #return (vSen @ self.V.T > 0.)
    return 1. / (1 + np.exp(-vSen @ self.V.T))

  def softmaxLayer(self, vHidden, debug=False, bias=False):
    prob = vHidden @ self.W.T 
    if debug:
      print('prob: ', prob)
    prob = np.exp(prob.T - logsumexp(prob, axis=1)).T
    return prob

  def align(self, aSen, vSen, unkProb=10e-12, debug=False):
    nState = len(vSen)
    T = len(aSen)
    scores = np.zeros((nState,))
    probs_z_given_y = self.softmaxLayer(self.hiddenLayer(vSen))
    
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
      scores = np.max(candidates, axis=0)
      if debug:
        print('self.init: ', self.init[nState])
        print('self.trans: ', self.trans[nState])
        print('backPtrs: ', backPointers[t])
        print('candidates: ', candidates)
      
      # XXX
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
    probs_z_given_y = self.softmaxLayer(self.hiddenLayer(vSen))
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
    
    np.save(fileName+'_softmaxweights.npy', self.W)
    np.save(fileName+'_hiddenweights.npy', self.V)
    
  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, isPhoneme=True, debug=False):
    f1 = open(filePrefix+'.txt', 'w')
    f2 = open(filePrefix+'_clusters.txt', 'w')
    aligns = []
    #if DEBUG:
    #  print(len(self.aCorpus))
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      alignment, alignProbs = self.align(aSen, vSen, debug=debug)
      cluster_assignment, clusterProbs = self.cluster(aSen, vSen, alignment)
      if DEBUG:
        print(aSen, vSen)
        print(type(alignment[1]))
      align_info = {
            'index': i, 
            'image_concepts': cluster_assignment,
            'alignment': alignment,
            'cluster_probs': clusterProbs, 
            'align_probs': alignProbs,
            'is_phoneme': isPhoneme
          }
      aligns.append(align_info)
      for a in alignment:
        f1.write('%d ' % a)
      f1.write('\n\n')
      for c in cluster_assignment:
        f2.write('%d ' % c)
      f2.write('\n\n')

    f1.close()
    f2.close()

    # Write to a .json file for evaluation
    with open(filePrefix+'.json', 'w') as f:
      json.dump(aligns, f, indent=4, sort_keys=True)            

if __name__ == '__main__':
  tasks = [3]
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
    modelConfigs = {'has_null': False, 'n_words': 3, 'learning_rate': 0.1, 'hidden_dim': 10, 'step_scale': 1.}
    model = ImagePhoneHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/jan_18_tiny_twolayer/tiny')
    model.trainUsingEM(100, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=30, T0=1., debug=False) 
    model.printAlignment('exp/jan_18_tiny_twolayer/tiny', debug=False)
  #-------------------------------------#
  # Image feature extraction for MSCOCO #
  #-------------------------------------#
  if 1 in tasks:
    featType = 'gaussian'    
    speechFeatureFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt'
    imageConceptFile = '../data/mscoco/trg_mscoco_subset_subword_level_power_law.txt'
    imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors_permute.npz'
    conceptIdxFile = 'exp/dec_30_mscoco/concept2idx.json'
    goldAlignmentFile = '../data/mscoco/mscoco_subset_synthetic_feature_vector_alignment.json'

    vCorpus = {}
    concept2idx = {}
    goldAlignments = {}
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
    permute = True
    centroids = 10 * np.random.normal(size=(nTypes, imgFeatDim)) 
     
    for ex, vSenStr in enumerate(vCorpusStr):
      N = len(vSenStr)
      if permute:
        alignment = np.random.permutation(np.arange(N))
      else:
        alignment = np.arange(N)

      if featType == 'one-hot':
        vSen = np.zeros((N, nTypes))
        for pos, i_a in enumerate(alignment.tolist()):
          vWord = vSenStr[i_a]
          vSen[pos, concept2idx[vWord]] = 1.
      elif featType == 'gaussian':
        vSen = np.zeros((N, imgFeatDim))
        for pos, i_a in enumerate(alignment.tolist()):
          vWord = vSenStr[i_a]
          vSen[pos] = centroids[concept2idx[vWord]] + 0.1 * np.random.normal(size=(imgFeatDim,))
        
      vCorpus['arr_'+str(ex)] = vSen
      goldAlignments['arr_'+str(ex)] = {'alignment': alignment.tolist()}

    np.savez(imageFeatureFile, **vCorpus)
    with open(conceptIdxFile, 'w') as f:
      json.dump(concept2idx, f, indent=4, sort_keys=True)
    with open(goldAlignmentFile, 'w') as f:
      json.dump(goldAlignments, f, indent=4, sort_keys=True)
  #--------------------------#
  # Word discovery on MSCOCO #
  #--------------------------#
  if 2 in tasks:      
    speechFeatureFile = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt'
    #speechFeatureFile = '../data/mscoco/trg_mscoco_subset_subword_level_power_law.txt'
    imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    #imageFeatureFile = 'mscoco_subset_subword_level_concept_vectors.npz'
    #imageFeatureFile = '../data/mscoco/mscoco_vgg_penult.npz'
    
    modelConfigs = {'has_null': False, 'n_words': 65, 'momentum': 0., 'learning_rate': 0.01, 'normalize_vfeat': False, 'step_scale': 0.001, 'hidden_dim': 10}
    modelName = 'exp/jan_20_mscoco_gaussian_momentum%.1f_lr%.2f_stepscale%.3f_twolayer/image_phone' % (modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale']) 
    print(modelName)
    #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
    model = ImagePhoneHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(30, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=100, T0=10., debug=True)
    #model.printAlignment('exp/dec_30_mscoco/image_phone_alignment', debug=False) 
    model.printAlignment(modelName+'_alignment', debug=False)
  #-----------------------------#
  # Word discovery on Flickr30k #
  #-----------------------------#
  if 3 in tasks:
    speechFeatureFile = '../data/flickr30k/phoneme_level/flickr30k_no_NULL.txt'
    #imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    #imageFeatureFile = '../data/mscoco/mscoco_subset_subword_level_concept_vectors.npz'
    imageFeatureFile = '../data/flickr30k/phoneme_level/flickr30k_no_NULL_vgg_penult.npz'
    modelConfigs = {'has_null': False, 'n_words': 100, 'momentum': 0.0, 'learning_rate': 0.01, 'normalize_vfeat': False, 'step_scale': 1., 'hidden_dim': 50}
    modelName = 'exp/jan_20_flickr_vgg16_momentum%.2f_lr%.5f_stepscale%d_two_layers/image_phone' % (modelConfigs['momentum'], modelConfigs['learning_rate'], modelConfigs['step_scale']) 
    print(modelName)
    #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_30_mscoco/image_phone') 
    model = ImagePhoneHMMDNNWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName=modelName)
    model.trainUsingEM(20, writeModel=True, debug=False)
    #model.simulatedAnnealing(numIterations=30, T0=50., debug=False)
    #model.printAlignment('exp/dec_30_mscoco/image_phone_alignment', debug=False) 
    model.printAlignment(modelName+'_alignment', debug=False) 
