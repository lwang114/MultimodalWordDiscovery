import numpy as np
import math
import json
from scipy.special import logsumexp

NULL = "NULL"
DEBUG = False

# TODO: Fix the normalization issue as in hmm_word_discoverer.py
# Audio-level word discovery model
# * The transition matrix is assumed to be Toeplitz 
class AudioHMMWordDiscoverer:
  def __init__(self, trainingCorpusFile, initProbFile=None, transProbFile=None, obsProbFile=None,
  modelName="audio_hmm_word_discoverer"):
  #def __init__(self, numMixtures, frameDim, sourceCorpusFile, targetCorpusFile, initProbFile=None, transProbFile=None, obsModelFile=None, modelName='audio_hmm_word_discoverer', maxLen=2000):
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
    self.initialize(trainingCorpusFile)
    #self.initialize(sourceCorpusFile, targetCorpusFile, maxLen);
    self.initProbFile = initProbFile
    self.transProbFile = transProbFile
    self.obsProbFile = obsProbFile
    #self.obsModelFile = obsModelFile
     
    self.fCorpus = self.fCorpus
    self.tCorpus = self.tCorpus 
    #self.obs_model = GMMWordDiscoverer(numMixtures, sourceCorpusFile, targetCorpusFile, maxLen=maxLen)
    print("Finish initialization of obs model")

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
      self.init[m] = np.log(1./m) * np.ones((m,))

    for m in self.lenProb:
      self.trans[m] = np.log(1./m) * np.ones((m, m))
 
  '''
  def initialize(self, fFileName, tFileName, maxLen):
    fp = open(tFileName)
    tCorpus = fp.read().split('\n')

    # XXX XXX    
    self.tCorpus = [[NULL] + tw.split() for tw in tCorpus[:2]]
    fp.close()
        
    fCorpus = np.load(fFileName) 
    # XXX XXX
    self.fCorpus = [fCorpus[k] for k in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))[:2]]
    self.fCorpus = [fSen[:maxLen] for fSen in self.fCorpus] 
        
    self.data_ids = [feat_id.split('_')[-1] for feat_id in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
    self.featDim = self.fCorpus[0].shape[1]
    
    # Initialize the transition probs uniformly 
    self.computeTranslationLengthProbabilities()
    
    for m in self.lenProb:
      self.init[m] = np.log(1. / m) * np.ones((m,))

    for m in self.lenProb:
      self.trans[m] = np.log(1. / m) * np.ones((m, m))
  '''

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
              self.obs[tw][fw] = 1.
    
      for tw in self.obs:
        totCount = sum(self.obs[tw].values())
        for fw in self.obs[tw].keys():
          #if DEBUG:
          #  if self.obs[tw][fw] > 1:
          #    print(self.obs[tw][fw])
          self.obs[tw][fw] = np.log(self.obs[tw][fw] / totCount) 
  
    '''if self.obsModelFile:
      self.initializeWordTranslationDensities(
            self.obsModelFile+'_mixture_priors.json',
            self.obsModelFile+'_translation_means.json',
            self.obsModelFile+'_translation_variances.json'  
            )
    '''
    
  def forward(self, eSen, fSen):
    T = len(fSen)
    nState = len(eSen)
    forwardProbs = -np.inf * np.ones((T, nState))
    for i in range(nState):
      #if fSen[0] in self.obs[eSen[i]]:
      #if DEBUG:
      #  print('init keys: ', self.init.keys())
      #  forwardProbs[0][i] = self.init[nState][i] * self.obs[eSen[i]][fSen[0]]
      #forwardProbs[0][i] = self.init[nState][i] + self.obs_model.logTransProb(fSen[0], eSen[i])
      forwardProbs[0][i] = self.init[nState][i] + self.obs[eSen[i]][fSen[0]]

    for t in range(T-1):
      obs_arr = np.array([self.obs[eSen[j]][fSen[t+1]] if fSen[t+1] in self.obs[eSen[j]] else 0 for j in range(nState)])  
      #forwardProbs[t+1] = self.trans[nState].T @ forwardProbs[t] * obs_arr
      #obs_arr = np.array([self.obs_model.logTransProb(fSen[t+1], tw) for tw in eSen])
      for j in range(nState):
        forwardProbs[t+1][j] = logsumexp(self.trans[nState][:, j] + forwardProbs[t]) + obs_arr[j]   
    
    assert not np.isnan(forwardProbs).any()
    return forwardProbs

  def backward(self, eSen, fSen):
    T = len(fSen)
    nState = len(eSen)
    backwardProbs = -np.inf * np.ones((T, nState))
    for i in range(nState):
      backwardProbs[T-1][i] = 0.

    for t in range(T-1, 0, -1):
      obs_arr = np.array([self.obs[eSen[j]][fSen[t]] if fSen[t] in self.obs[eSen[j]] else 0 for j in range(nState)])
      #backwardProbs[t-1] = self.trans[nState] @ (backwardProbs[t] * obs_arr)
      #obs_arr = np.array([self.obs_model.logTransProb(fSen[t], tw) for tw in eSen])
      for j in range(nState):
        backwardProbs[t-1][j] = logsumexp(self.trans[nState][j] + backwardProbs[t] + obs_arr)
     
    assert not np.isnan(backwardProbs).any()
    return backwardProbs  

  def updateInitialCounts(self, forwardProbs, backwardProbs, eSen, fSen):
    #assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all() 
    nState = len(eSen)
    T = len(fSen)
    # Update the initial prob  
    initExpCounts = -np.inf * np.ones((nState,))  
    for i in range(nState):
      initExpCounts[i] = logsumexp(forwardProbs[:, i] + backwardProbs[:, i])
    
    assert not np.isnan(initExpCounts).any()
    return initExpCounts

  def updateTransitionCounts(self, forwardProbs, backwardProbs, eSen, fSen):
    nState = len(eSen)
    T = len(fSen)
    transExpCounts = -np.inf * np.ones((nState, nState))
    # Update the transition probs
    for t in range(T-1):
      obs_arr = np.array([self.obs[eSen[j]][fSen[t+1]] if fSen[t+1] in self.obs[eSen[j]] else 0 for j in range(nState)])
      #transExpCounts += np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans  
      #transExpCount = np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans[nState]
      #obs_arr = np.array([self.obs_model.logTransProb(fSen[t+1], tw) for tw in eSen])
      transExpCount = np.tile(forwardProbs[t], (nState, 1)).T + self.trans[nState] + obs_arr + backwardProbs[t+1]
      # Maintain Toeplitz assumption
      # TODO: Make this more efficient
      transJumpCount = {}
      transJumpCounts = {}
      for s in range(nState):
        for next_s in range(nState):
          if next_s - s not in transJumpCount:
            #if DEBUG:
            #  print('new jump: ', next_s - s) 
            transJumpCount[next_s - s] = [transExpCount[s, next_s]]
          else:
            transJumpCount[next_s - s].append(transExpCount[s, next_s])

    for s in range(nState):
      for next_s in range(nState):
        transJumpCounts[next_s - s] = logsumexp(np.asarray(transJumpCount[next_s - s]))

    for s in range(nState):
      for next_s in range(nState):
        transExpCounts[s][next_s] = transJumpCounts[next_s - s]

    #if DEBUG:
    #  print('product: ', np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans[nState])
    #if DEBUG:
    #  print('transExpCounts: ', transExpCounts)
    
    assert not np.isnan(transExpCounts).any()
    return transExpCounts
  
  def updateObservationCounts(self, forwardProbs, backwardProbs, eSen, fSen):
    #assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all()
    # Update observation probs
    nState = len(eSen)
    newObsCounts = {tw: {} for tw in self.obs}
    statePosteriors = forwardProbs + backwardProbs
    normFactor = logsumexp(statePosteriors.flatten(order='C'))
    statePosteriors = (statePosteriors.T - normFactor).T
    
    for t, w in enumerate(fSen): 
      for i in range(nState):
        if w not in newObsCounts[eSen[i]]:
          newObsCounts[eSen[i]][w] = []
        newObsCounts[eSen[i]][w].append(statePosteriors[t][i]) #self.obs[eSen[i]][w]
    
    return newObsCounts

  '''
  def updateObservationCounts(self, forwardProbs, backwardProbs, countByObsModel, eSen, fSen): 
    nState = len(eSen)
    statePosteriors = forwardProbs + backwardProbs
    normFactor = logsumexp(statePosteriors.flatten(order='C'))
    statePosteriors = (statePosteriors.T - normFactor).T
    newObsCounts = {k: -np.inf * np.ones(count.shape) for k, count in countByObsModel.items()}    

    for t, fw in enumerate(fSen):
      for i, tw in enumerate(sorted(eSen)):
        tKey = "_".join([str(i), tw])
        newObsCounts[tKey] = statePosteriors[:, i] + countByObsModel[tKey]
        assert not np.isnan(newObsCounts[tKey]).any()

    return newObsCounts
  '''

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
      likelihood = logsumexp(forwardProb[-1])
      ll += likelihood
    return ll / len(self.tCorpus)

  def trainUsingEM(self, numIterations=30, writeModel=False):
    if writeModel:
      self.printModel('initial_model.txt')
 
    self.initializeModel()   
    #self.obs_model.computeExpectedCounts()
    initCounts = {m: [] for m in self.lenProb}
    transCounts = {m: [] for m in self.lenProb}
    #obsCounts = []
    obsCounts = {tw: {fw: [] for fw in self.obs[tw]} for tw in self.obs}  
 
    for epoch in range(numIterations): 
      #AvgLogProb = self.computeAvgLogLikelihood()
      for i_ex, (eSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        forwardProbs = self.forward(eSen, fSen)
        backwardProbs = self.backward(eSen, fSen) 
        initCounts[len(eSen)].append(self.updateInitialCounts(forwardProbs, backwardProbs, eSen, fSen))
        transCounts[len(eSen)].append(self.updateTransitionCounts(forwardProbs, backwardProbs, eSen, fSen))
        #obsCounts.append(self.updateObservationCounts(forwardProbs, backwardProbs, self.obs_model.alignProb[i_ex], eSen, fSen))
        obsCount = self.updateObservationCounts(forwardProbs, backwardProbs, eSen, fSen)
        for tw in obsCount:
          for fw in obsCount[tw]:
            if DEBUG:
              print(obsCount[tw][fw])
            obsCounts[tw][fw].append(logsumexp(np.asarray(obsCount[tw][fw])))

        if DEBUG:
          if i_ex == 0:
            #print("forwardProbs, backwardProbs: ", forwardProbs, backwardProbs)
            #print("transCount: ", self.updateTransitionCounts(forwardProbs, backwardProbs, eSen, fSen))
            print("initCount: ", self.updateInitialCounts(forwardProbs, backwardProbs, eSen, fSen))
            #print("obsCount: ", self.updateObservationCounts(forwardProbs, backwardProbs, self.obs_model.alignProb[i_ex], eSen, fSen))

      # Update the parameters of the observation model
      #self.obs_model.alignProb = obsCounts
      #self.obs_model.updateTranslationDensities()

      # Normalize
      for m in self.lenProb:
        for s in range(m):
          #print(np.ascontiguousarray(initCounts[m])[:, s].flags['C_CONTIGUOUS'])
          counts_s = np.asarray([count[s] for count in initCounts[m]])
          self.init[m][s] = logsumexp(counts_s) 
        normFactor = logsumexp(np.asarray(initCounts[m]).flatten(order='C'))
        self.init[m] -= normFactor
        if DEBUG:
          print("np.sum(self.init): ", np.sum(np.exp(self.init[m])))

        for r in range(m):
          for s in range(m):
            counts_r_s = np.asarray([np.array(count[r][s]) for count in transCounts[m]]) 
            self.trans[m][r, s] = logsumexp(counts_r_s)
          normFactor = logsumexp(np.asarray([count[r] for count in transCounts[m]]).flatten(order='C'))
          self.trans[m][r] -= normFactor 
          if DEBUG:
            print("np.sum(self.trans[row]): ", np.sum(np.exp(self.trans[m][r])))

        for tw in self.obs:
          obs_count_tw = []
          for fw in obsCounts[tw]:
            if DEBUG:
              print("tw, fw, obsCounts[tw][fw]: ", tw, fw, obsCounts[tw][fw])
            obs_count_tw.extend(obsCounts[tw][fw])
         
          if DEBUG:
            print("np.exp(normFactor): ", np.exp(np.asarray(obs_count_tw).flatten(order='C'))) 
          normFactor = logsumexp(np.asarray(obs_count_tw).flatten(order='C'))
          if normFactor == 0:
            if DEBUG:
              print('norm factor for the obs is 0: potential bug')
            self.obs[tw][fw] = self.obs[tw][fw]

          for fw in obsCounts[tw]:
            self.obs[tw][fw] = logsumexp(np.asarray(obsCounts[tw][fw])) - normFactor

      print('Epoch', epoch, 'Average Log Likelihood:', self.computeAvgLogLikelihood())  
      if writeModel:
        self.printModel(self.modelName+'model_iter='+str(epoch))

  # TODO
  def align(self, fSen, eSen, unkProb=10e-12):
    nState = len(eSen)
    T = len(fSen)
    scores = np.zeros((nState,))
    backPointers = np.zeros((T, nState), dtype=int)
    for i in range(nState):
      scores[i] = self.init[nState][i] + self.obs[eSen[i]][fSen[0]] 
      #scores[i] = self.init[nState][i] + self.obs_model.logTransProb(fSen[0], eSen[i]) 

    alignProbs = [] 
    for t, fw in enumerate(fSen[1:]):
      obs_arr = np.array([self.obs[eSen[i]][fw] if fw in self.obs[eSen[i]] else unkProb for i in range(nState)])
      #obs_arr = np.array([self.obs_model.logTransProb(fw, tw) for tw in eSen]) 
      #candidates = np.tile(scores, (nState, 1)).T * self.trans[nState] * obs_arr
      
      candidates = np.tile(scores, (nState, 1)).T + self.trans[nState] + obs_arr
      backPointers[t+1] = np.argmax(candidates, axis=0)
      scores = np.max(candidates, axis=0)
      alignProbs.append(scores.tolist())
      
      #if DEBUG:
      #  print(scores)
    
    curState = np.argmax(scores)
    bestPath = [int(curState)]
    for t in range(T-1, 0, -1):
      #if DEBUG:
      #  print('curState: ', curState)
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

    #self.obs_model.printModel(fileName+'_obs_model')
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
      #if DEBUG:
      #  print(fSen, tSen)
      #  print(type(alignment[1]))
      #align_info = {
      #      'index': self.data_ids[i],
      align_info = {
            'index': i,
            'image_concepts': tSen,
            'alignment': alignment,
            'align_probs': alignProbs,
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
  trainingCorpusFile = 'test_translation.txt' 
  #'../data/flickr30k/phoneme_level/flickr30k.txt'
  #initProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_initialprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_initialprobs.txt'
  #transProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_transitionprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_transitionprobs.txt'
  #obsProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_observationprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_observationprobs.txt'
  sourceCorpusFile = "../data/flickr30k/audio_level/flickr_mfcc_cmvn_htk.npz"
  targetCorpusFile = "../data/flickr30k/audio_level/flickr_bnf_all_trg.txt"
  model = AudioHMMWordDiscoverer(1, 12, sourceCorpusFile, targetCorpusFile, modelName='test_audio_hmm', maxLen=20)
  #model = AudioHMMWordDiscoverer(trainingCorpusFile, modelName='test_audio_hmm')
  model.trainUsingEM(10, writeModel=True)
  model.printAlignment('alignment')

  #model = HMMWordDiscoverer(trainingCorpusFile, modelName='A')
  #model.trainUsingEM(10)  
