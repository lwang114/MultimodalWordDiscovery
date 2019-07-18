import numpy as np
import math
import json

NULL = "NULL"
DEBUG = False

# TODO: Incorporate this under HMM class
# A word discovery model based on Vogel et. al. 1996
# * The transition matrix is assumed to be Toeplitz 
class HMMWordDiscoverer:
  def __init__(self, trainingCorpusFile, initProbFile=None, transProbFile=None, obsProbFile=None, modelName='hmm_word_discoverer'):
    self.modelName = modelName 
    # Initialize data structures for storing training data
    self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

    self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences
    
    self.init = {}
    self.obs = {}                     # obs[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.avgLogTransProb = float('-inf')
     
    # Read the corpus
    self.initialize(trainingCorpusFile);
    self.initProbFile = initProbFile
    self.transProbFile = transProbFile
    self.obsProbFile = obsProbFile
     
    self.fCorpus = self.fCorpus
    self.tCorpus = self.tCorpus 
  
  def initialize(self, fileName):
    f = open(fileName)

    i = 0
    j = 0;
    tTokenized = ();
    fTokenized = ();
    for s in f:
        if i == 0:
            tTokenized = s.split() #word_tokenize(s)
            # Add null word in position zero
            tTokenized.insert(0, NULL)
            self.tCorpus.append(tTokenized)
        elif i == 1:
            fTokenized = s.split()
            self.fCorpus.append(fTokenized)
        else:
            i = -1
            j += 1
        i +=1
    f.close()
    
    # Initialize the transition probs uniformly 
    self.computeTranslationLengthProbabilities()
    
    for m in self.lenProb:
      self.init[m] = 1 / m * np.ones((m,))

    for m in self.lenProb:
      self.trans[m] = 1 / m * np.ones((m, m))
  
  # Set initial values for the translation probabilities p(f|e)
  def initializeModel(self):
    self.obs = {}
    if self.initProbFile:
      f = open(self.initProbFile)
      for line in f:
        m, s, prob = line.split()
        self.init[int(m)][int(s)] = float(prob)

    if self.transProbFile:
      f = open(self.transProbFile)
      for line in f:
        m, cur_s, next_s, prob = line.split()
        self.trans[int(m)][int(cur_s)][int(next_s)] = float(prob)     

    if self.obsProbFile:
      f = open(self.obsProbFile)
      for line in f:
        tw, fw, prob = line.strip().split()
        if tw not in self.obs.keys():
          self.obs[tw] = {}
        self.obs[tw][fw] = float(prob)
         
      f.close()
    else:
      for ts, fs in zip(self.tCorpus, self.fCorpus):
        for tw in ts:
          for fw in fs:
            if tw not in self.obs.keys():
              self.obs[tw] = {}  
            if fw not in self.obs[tw].keys():
              self.obs[tw][fw] = 1
    
      for tw in self.obs:
        totCount = sum(self.obs[tw].values())
        for fw in self.obs[tw].keys():
          #if DEBUG:
          #  if self.obs[tw][fw] > 1:
          #    print(self.obs[tw][fw])
          self.obs[tw][fw] = self.obs[tw][fw] / totCount 
 
  def forward(self, eSen, fSen):
    T = len(fSen)
    nState = len(eSen)
    forwardProbs = np.zeros((T, nState))
    for i in range(nState):
      if fSen[0] in self.obs[eSen[i]]:
        if DEBUG:
          print('init keys: ', self.init.keys())
        forwardProbs[0][i] = self.init[nState][i] * self.obs[eSen[i]][fSen[0]]
    
    # Implement scaling if necessary
    for t in range(T-1):
      obs_arr = np.array([self.obs[eSen[j]][fSen[t+1]] if fSen[t+1] in self.obs[eSen[j]] else 0 for j in range(nState)])  
      forwardProbs[t+1] = self.trans[nState].T @ forwardProbs[t] * obs_arr
    
    return forwardProbs

  def backward(self, eSen, fSen):
    T = len(fSen)
    nState = len(eSen)
    backwardProbs = np.zeros((T, nState))
    for i in range(nState):
      backwardProbs[T-1][i] = 1

    for t in range(T-1, 0, -1):
      obs_arr = np.array([self.obs[eSen[j]][fSen[t]] if fSen[t] in self.obs[eSen[j]] else 0 for j in range(nState)])
      backwardProbs[t-1] = self.trans[nState] @ (backwardProbs[t] * obs_arr)
        
    return backwardProbs  

  def updateInitialCounts(self, forwardProbs, backwardProbs, eSen, fSen):
    assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all() 
    nState = len(eSen)
    T = len(fSen)
    # Update the initial prob  
    initExpCounts = np.zeros((nState,))  
    for t in range(T):
      initExpCounts += forwardProbs[t] * backwardProbs[t]
    
    return initExpCounts

  def updateTransitionCounts(self, forwardProbs, backwardProbs, eSen, fSen):
    nState = len(eSen)
    T = len(fSen)
    transExpCounts = np.zeros((nState, nState))
    # Update the transition probs
    for t in range(T-1):
      obs_arr = np.array([self.obs[eSen[j]][fSen[t+1]] if fSen[t+1] in self.obs[eSen[j]] else 0 for j in range(nState)])
      #transExpCounts += np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans  
      transExpCount = np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans[nState]
      # Maintain Toeplitz assumption
      # TODO: Make this more efficient
      transJumpCount = {}
      for s in range(nState):
        for next_s in range(nState):
          if next_s - s not in transJumpCount:
            if DEBUG:
              print('new jump: ', next_s - s) 
            transJumpCount[next_s - s] = transExpCount[s][next_s]
          else:
            transJumpCount[next_s - s] += transExpCount[s][next_s]

      for s in range(nState):
        for next_s in range(nState):
          transExpCounts[s][next_s] += transJumpCount[next_s - s]

      if DEBUG:
        print('forward prob, obs, backward prob: ', np.tile(forwardProbs[t], (nState, 1)), obs_arr, backwardProbs[t])
        print('product: ', np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans[nState])
    
    if DEBUG:
      print('transExpCounts: ', transExpCounts)

    return transExpCounts
     
  def updateObservationCounts(self, forwardProbs, backwardProbs, eSen, fSen):
    assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all()
    # Update observation probs
    nState = len(eSen)
    newObsCounts = {tw: {fw: 0. for fw in self.obs[tw]} for tw in self.obs}
    statePosteriors = ((forwardProbs * backwardProbs).T / np.sum(forwardProbs * backwardProbs, axis=1)).T
    if DEBUG:
      print('statePosteriors: ', statePosteriors)
    
    for t, w in enumerate(fSen): 
      for i in range(nState):
        if w not in newObsCounts[eSen[i]]:
          newObsCounts[eSen[i]][w] = 0.
        newObsCounts[eSen[i]][w] += statePosteriors[t][i] #self.obs[eSen[i]][w]
    
    return newObsCounts

  # Compute translation length probabilities q(m|n)
  def computeTranslationLengthProbabilities(self, smoothing=None):
      # Implement this method
      #pass        
      #if DEBUG:
      #  print(len(self.tCorpus))
      for ts, fs in zip(self.tCorpus, self.fCorpus):
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
      
      # TODO: Kneser-Ney smoothing
      for tl in self.lenProb.keys():
        totCount = sum(self.lenProb[tl].values())  
        for fl in self.lenProb[tl].keys():
          self.lenProb[tl][fl] = self.lenProb[tl][fl] / totCount 

  def computeAvgLogLikelihood(self):
    ll = 0.
    for tSen, fSen in zip(self.tCorpus, self.fCorpus):
      forwardProb = self.forward(tSen, fSen)
      #backwardProb = self.backward(tSen, fSen)
      likelihood = np.sum(forwardProb[-1])
      ll += math.log(likelihood)
    return ll / len(self.tCorpus)

  def trainUsingEM(self, numIterations=50, writeModel=False, convergenceEpsilon=0.01):
    if writeModel:
      self.printModel('initial_model.txt')
 
    self.initializeModel()
    initCounts = {m: np.zeros((m,)) for m in self.lenProb}
    transCounts = {m: np.zeros((m, m)) for m in self.lenProb}
    obsCounts = {tw: {fw: 0. for fw in self.obs[tw]} for tw in self.obs}  
    
    for epoch in range(numIterations): 
      AvgLogProb = self.computeAvgLogLikelihood()
      print('Epoch', epoch, 'Average Log Likelihood:', self.computeAvgLogLikelihood())  
      
      for eSen, fSen in zip(self.tCorpus, self.fCorpus):
          forwardProbs = self.forward(eSen, fSen)
          backwardProbs = self.backward(eSen, fSen) 
          initCounts[len(eSen)] += self.updateInitialCounts(forwardProbs, backwardProbs, eSen, fSen)
          transCounts[len(eSen)] += self.updateTransitionCounts(forwardProbs, backwardProbs, eSen, fSen)
          obsCountsInc = self.updateObservationCounts(forwardProbs, backwardProbs, eSen, fSen)
          for tw in obsCountsInc:
            for fw in obsCountsInc[tw]:
              if fw not in obsCounts[tw]:
                obsCounts[tw][fw] = obsCountsInc[tw][fw]
              else:
                obsCounts[tw][fw] += obsCountsInc[tw][fw]

      # Normalize
      for m in self.lenProb:
        self.init[m] = initCounts[m] / np.sum(initCounts[m]) 
  
      for m in self.lenProb:
        totCounts = np.sum(transCounts[m], axis=1)
        for s in range(m):
          if totCounts[s] == 0:
            # Not updating the transition arc if it is not used          
            self.trans[m][s] = self.trans[m][s]
          else:
            self.trans[m][s] = transCounts[m][s] / totCounts[s]
      
      for tw in self.obs:
        normFactor = sum(obsCounts[tw].values())
        if normFactor == 0:
          if DEBUG:
            print('norm factor for the obs is 0: potential bug')
          self.obs[tw][fw] = self.obs[tw][fw]

        for fw in obsCounts[tw]:
          self.obs[tw][fw] = obsCounts[tw][fw] / normFactor

      if writeModel:
        self.printModel(self.modelName + '_iter='+str(epoch)+'.txt')
    
  def align(self, fSen, eSen, unkProb=10e-12):
    nState = len(eSen)
    T = len(fSen)
    scores = np.zeros((nState,))
    backPointers = np.zeros((T, nState), dtype=int)
    for i in range(nState):
      scores[i] = self.init[nState][i] * self.obs[eSen[i]][fSen[0]] 
    
    alignProbs = [] 
    for t, fw in enumerate(fSen[1:]):
      obs_arr = np.array([self.obs[eSen[i]][fw] if fw in self.obs[eSen[i]] else unkProb for i in range(nState)])
      candidates = np.tile(scores, (nState, 1)).T * self.trans[nState] * obs_arr
      backPointers[t+1] = np.argmax(candidates, axis=0)
      scores = np.max(candidates, axis=0)
      
      alignProbs.append((scores / np.sum(scores)).tolist())
      
      if DEBUG:
        print(scores)
    
    curState = np.argmax(scores)
    bestPath = [int(curState)]
    for t in range(T-1, 0, -1):
      if DEBUG:
        print('curState: ', curState)
      curState = backPointers[t, curState]
      bestPath.append(int(curState))
    
    return bestPath[::-1], alignProbs
      
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

    obsFile = open(fileName+'_observationprobs.txt', 'w')
     
    for tw in sorted(self.obs):
      for fw in sorted(self.obs[tw]):
        obsFile.write('%s\t%s\t%f\n' % (tw, fw, self.obs[tw][fw]))
    obsFile.close()
   
  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, isPhoneme=True):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    if DEBUG:
      print(len(self.fCorpus))
    for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
      alignment, alignProbs = self.align(fSen, tSen)
      if DEBUG:
        print(fSen, tSen)
        print(type(alignment[1]))
      align_info = {
            'index': i,
            'image_concepts': tSen, 
            'caption': fSen,
            'alignment': alignment,
            'align_probs': alignProbs,
            'is_phoneme': isPhoneme
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
  trainingCorpusFile = 'test_translation.txt' 
  #'../data/flickr30k/phoneme_level/flickr30k.txt'
  initProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_initialprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_initialprobs.txt'
  transProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_transitionprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_transitionprobs.txt'
  obsProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_observationprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_observationprobs.txt'

  model = HMMWordDiscoverer(trainingCorpusFile, modelName='A')
  #model = HMMWordDiscoverer(trainingCorpusFile, initProbFile, transProbFile, obsProbFile, modelName='A')
  model.trainUsingEM(50, writeModel=True)
  model.printAlignment('alignment')
