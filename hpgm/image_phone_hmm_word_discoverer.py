import numpy as np
import math
import json
import time

NULL = "NULL"
DEBUG = False

# A word discovery model using image regions and phones
# * The transition matrix is assumed to be Toeplitz 
class ImagePhoneHMMWordDiscoverer:
  def __init__(self, speechFeatureFile, imageFeatureFile, modelConfigs, initProbFile=None, transProbFile=None, obsProbFile=None, modelName='image_phone_hmm_word_discoverer'):
    self.modelName = modelName 
    # Initialize data structures for storing training data
    self.aCorpus = []                   # aCorpus is a list of acoustic features

    self.vCorpus = []                   # vCorpus is a list of image posterior features (e.g. VGG softmax)
    self.hasNull = modelConfigs.get('has_null', False)
    self.nWords = modelConfigs.get('n_words', 66) 
    self.momentum = modelConfigs.get('momentum', 1.)
    self.lr = modelConfigs.get('learning_rate', 1e-2)

    self.init = {}
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.obs = None                 # obs[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
    self.avgLogTransProb = float('-inf')
     
    # Read the corpus
    self.readCorpus(speechFeatureFile, imageFeatureFile);
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
    vCorpus = [vNpz[k] for k in sorted(vNpz.keys(), key=lambda x:int(x.split('_')[-1]))]
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
    print("Total number of word types: ", self.nWords)
    
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

    self.W = np.random.normal(size=(self.nWords, self.imageFeatDim))  
    print("Finish initialization after %0.3f s" % (time.time() - begin_time))

  def trainUsingEM(self, numIterations=80, writeModel=False, convergenceEpsilon=0.01, debug=False):
    self.initializeModel()
    if writeModel:
      self.printModel('initial_model.txt')
    
    for epoch in range(numIterations): 
      initCounts = {m: np.zeros((m,)) for m in self.lenProb}
      transCounts = {m: np.zeros((m, m)) for m in self.lenProb}
      phoneCounts = np.zeros((self.nWords, self.audioFeatDim))      
      conceptCounts = [np.zeros((vSen.shape[0], self.nWords)) for vSen in self.vCorpus]
      AvgLogProb = self.computeAvgLogLikelihood()
      print('Epoch', epoch, 'Average Log Likelihood:', self.computeAvgLogLikelihood())  
      
      for ex, (vSen, aSen) in enumerate(zip(self.vCorpus, self.aCorpus)):
        forwardProbs = self.forward(vSen, aSen, debug=debug)
        backwardProbs = self.backward(vSen, aSen, debug=debug) 
        #print('forward prob, backward prob', forwardProbs, backwardProbs)
        initCounts[len(vSen)] += np.sum(self.updateInitialCounts(forwardProbs, backwardProbs, vSen, aSen), axis=-1)
        transCounts[len(vSen)] += np.sum(self.updateTransitionCounts(forwardProbs, backwardProbs, vSen, aSen), axis=-1)
        stateCounts = self.updateStateCounts(forwardProbs, backwardProbs)
        phoneCounts += np.sum(stateCounts, axis=1).T @ aSen
        conceptCounts[ex] += np.sum(stateCounts, axis=0)

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
      
      normFactor = np.sum(phoneCounts, axis=-1) 
      self.obs = (phoneCounts.T / normFactor).T
      # TODO: merge this into the training iterations
      self.updateSoftmaxWeight(conceptCounts) 

      if epoch % 10 == 0:
        self.lr /= 10

        if writeModel:
          self.printModel(self.modelName + '_iter='+str(epoch)+'.txt')

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
    if debug:
      print('self.lenProb.keys: ', self.lenProb.keys())
      print('init keys: ', self.init.keys())
      print('nState: ', nState)
    
    probs_z_given_y = self.softmaxLayer(vSen) 
    
    forwardProbs[0] = np.tile(self.init[nState][:, np.newaxis], (1, self.nWords)) * probs_z_given_y * (self.obs @ aSen[0])
    for t in range(T-1):
      probs_x_t_z_given_y = probs_z_given_y * (self.obs @ aSen[t])
      if debug:
        print('probs_x_t_z_given_y.shape: ', probs_x_t_z_given_y.shape)
      # Compute the diagonal term
      forwardProbs[t+1] += (np.diag(self.trans[nState]) * forwardProbs[t].T).T * probs_x_t_z_given_y 
      # Compute the off-diagonal term 
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
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
      backwardProbs[t-1] += (np.diag(self.trans[nState]) * backwardProbs[t].T).T * prob_x_t_z_given_y 
      trans_off_diag = self.trans[nState] - np.diag(np.diag(self.trans[nState]))
      backwardProbs[t-1] += trans_off_diag @ (np.sum(backwardProbs[t], axis=-1) * prob_x_t_z_given_y.T).T
        
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
      initExpCounts += np.sum(forwardProbs[t] * backwardProbs[t], axis=-1) / np.sum(forwardProbs[t] * backwardProbs[t])
      if debug:
        print('forwardProbs, backwardProbs: ', forwardProbs[t], backwardProbs[t])
    
    if debug:
      print('initExpCounts 1: ', initExpCounts)

    initExpCounts = np.zeros((nState,))   
    initExpCount = forwardProbs * backwardProbs
    initExpCount /= np.sum(initExpCount, keepdims=True, axis=1) 
    initExpCounts = np.sum(initExpCount, axis=0) 
    if debug:
      print('initExpCounts 2: ', initExpCounts)

    return initExpCounts

  # Inputs:
  # ------
  #   forwardProbs: Tx x Ty x K matrix storing p(z_i, i_t, x_1:t|y)
  #   backwardProbs: Tx x Ty x K matrix storing p(x_t+1:Tx|z_i, i_t, y)   
  #   vSen: Ty x Dx matrix storing the image feature (e.g., VGG 16 hidden activation)
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
    probs_x_t_given_y_i = (probs_z_given_y @ (self.obs @ aSen.T)).T  
    for t in range(T-1):
      #transExpCounts += np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans  
      alpha = np.tile(np.sum(forwardProbs[t], axis=-1), (nState, 1))
      beta_next = np.sum(backwardProbs[t+1], axis=-1)
      transExpCount = alpha.T * (probs_x_t_given_y_i[t+1] * beta_next) * self.trans[nState]
      #print("transExpCount: ", transExpCount)
      transExpCount /= np.sum(transExpCount)

      # Maintain Toeplitz assumption
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

      if debug:
        print('forward prob, obs, backward prob: ', np.tile(forwardProbs[t], (nState, 1)), obs_arr, backwardProbs[t])
        print('product: ', np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans[nState])
    
    if debug:
      print('transExpCounts: ', transExpCounts)

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
    assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all()
    T = forwardProbs.shape[0]
    nState = forwardProbs.shape[1]
    normFactor = np.sum(np.sum(forwardProbs * backwardProbs, axis=-1), axis=-1)
    newStateCounts = np.transpose(np.transpose(forwardProbs * backwardProbs, (1, 2, 0)) / normFactor, (2, 0, 1)) 
   
    return newStateCounts

  # Inputs:
  # ------
  #   conceptCounts: a list of Ty x K matrices storing p(z_i|x, y) for each utterances
  #   numGDIterations: int, number of gradient descent iterations   
  #
  # Outputs:
  # -------
  #   None
  def updateSoftmaxWeight(self, conceptCounts):
    dW = np.zeros((self.nWords, self.imageFeatDim))
    
    for vSen, conceptCount in zip(self.vCorpus, conceptCounts):
      z_prob = self.softmaxLayer(vSen) 
      dWs = self.lr * np.tile(conceptCount[:, :, np.newaxis] * (1. - z_prob[:, :, np.newaxis]), (1, 1, self.imageFeatDim)) * vSen[:, np.newaxis, :]
      dW += np.sum(dWs, axis=0)

    self.W = self.momentum * self.W + dW

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
      likelihood = np.sum(forwardProb[-1])
      #print(likelihood)
      ll += math.log(likelihood)
    return ll / len(self.vCorpus)
  
  def softmaxLayer(self, vSen):
    prob = np.exp(vSen @ self.W.T)
    prob = (prob.T / np.sum(prob, axis=1)).T
    return prob

  def align(self, aSen, vSen, unkProb=10e-12):
    nState = len(vSen)
    T = len(aSen)
    scores = np.zeros((nState,))
    probs_z_given_y = self.softmaxLayer(vSen)
    
    backPointers = np.zeros((T, nState), dtype=int)
    probs_x_given_y = (probs_z_given_y @ (self.obs @ aSen.T)).T    
    scores = self.init[nState] * probs_x_given_y[0]
    
    alignProbs = [] 
    for t, phone in enumerate(aSen[1:]):
      candidates = np.tile(scores, (nState, 1)).T * self.trans[nState] * probs_x_given_y[t]
      backPointers[t+1] = np.argmax(candidates, axis=0)
      scores = np.max(candidates, axis=0)
      
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
    

  # Write the predicted alignment to file
  def printAlignment(self, filePrefix, isPhoneme=True):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    #if DEBUG:
    #  print(len(self.aCorpus))
    for i, (aSen, vSen) in enumerate(zip(self.aCorpus, self.vCorpus)):
      alignment, alignProbs = self.align(aSen, vSen)
      if DEBUG:
        print(aSen, vSen)
        print(type(alignment[1]))
      align_info = {
            'index': i,
            'image_concepts': ['NULL']+['0']*len(vSen),
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
  tasks = [0]
  if 0 in tasks:
    eps = 0.
    speechFeatureFile = 'tiny.txt'
    imageFeatureFile = 'tiny.npz'
    image_feats = {'0_1':np.array([[1-eps, eps/2., eps/2.], [eps/2., 1.-eps, eps/2.]]), '0_2':np.array([[eps/2., 1.-eps, eps/2.], [eps/2., eps/2., 1.-eps]]), '0_3':np.array([[eps/2., eps/2., 1.-eps], [1.-eps, eps/2., eps/2.]])}
    audio_feats = '0 1\n1 2\n2 0'
    with open('tiny.txt', 'w') as f:
      f.write(audio_feats)
    np.savez('tiny.npz', **image_feats)
    modelConfigs = {'has_null': False, 'n_words': 3}
    model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelConfigs, modelName='exp/dec_11_tiny/tiny')
    model.trainUsingEM(20, writeModel=True, debug=True)
    model.printAlignment('exp/dec_11_tiny/tiny')
  
  #if 1 in tasks:
  # TODO
  #model = ImagePhoneHMMWordDiscoverer(speechFeatureFile, imageFeatureFile, modelName='image_phone')
  #speechFeatureFile = '../data/flickr30k/sensory_level/flickr30k_vgg_softmax_captions.txt' #'test_translation.txt' 
  #imageFeatureFile = '../data/flickr30k/sensory_level/flickr30k_vgg_softmax.npz' #'test_translation.txt' 
  #'../data/flickr30k/phoneme_level/flickr30k.txt'
  #initProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_initialprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_initialprobs.txt'
  #transProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_transitionprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_transitionprobs.txt'
  #obsProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_observationprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_observationprobs.txt'
  

