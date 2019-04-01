import numpy as np
import math
import json

DEBUG = False
EMPTY = '</eps>'
class HiddenMarkovModel(object):
  def __init__(self, trainCorpus):
    pass

  def forward():
    raise NotImplementedError

  def backward():
    raise NotImplementedError

  def trainUsingEM(self, numIterations=10, writeModel=False, convergenceEpsilon=0.01):
    raise NotImplementedError


class DiscreteHMM(HiddenMarkovModel):
  def __init__(self, nState, trainCorpus, modelName='dhmm', emptySymbol=EMPTY):
    super(DiscreteHMM, self).__init__(trainCorpus)
    self.modelName = modelName
    # Assumed to be properly tokenized
    self.trainCorpus = trainCorpus
    self.init = np.zeros((nState,))
    self.init[0] = 1.
    self.trans = np.zeros((nState, nState))
    self.obs = [{} for i in range(nState)]
    self.empty = emptySymbol
    self.nState = self.init.shape[0]
    self.initialize()
  
  # TODO: Load pretrained weights to the model
  def initialize(self):
    # Initialize the transition probs with left-to-right uniform model
    n = self.init.shape[0]
    for i in range(n):
      for j in range(i, n):
        self.trans[i][j] = 1 / (n - i)

    for i in range(n):
      for sent in self.trainCorpus:
        for w in sent[i:]:
          if not w in self.obs[i]:
            self.obs[i][w] = 1
  
      totCount = sum(self.obs[i].values())  
      for w in self.obs[i]:
        self.obs[i][w] /= totCount   
             
  def forward(self, sent):
    T = len(sent)
    nState = self.init.shape[0]
    forwardProbs = np.zeros((T, nState))
    for i in range(nState):
      if sent[0] in self.obs[i]:
        forwardProbs[0][i] = self.init[i] * self.obs[i][sent[0]]
    
    # Implement scaling if necessary
    for t in range(T-1):
      obs_arr = np.array([self.obs[j][sent[t+1]] if sent[t+1] in self.obs[j] else 0 for j in range(nState)])  
      forwardProbs[t+1] = self.trans.T @ forwardProbs[t] * obs_arr
    
    return forwardProbs

  def backward(self, sent):
    T = len(sent)
    nState = self.init.shape[0] 
    backwardProbs = np.zeros((T, nState))
    for i in range(nState):
      backwardProbs[T-1][i] = 1

    for t in range(T-1, 0, -1):
      obs_arr = np.array([self.obs[j][sent[t]] if sent[t] in self.obs[j] else 0 for j in range(nState)])
      backwardProbs[t-1] = self.trans @ (backwardProbs[t] * obs_arr)
        
    return backwardProbs  

  def updateInitialCounts(self, forwardProbs, backwardProbs, sent):
    assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all() 
    nState = self.init.shape[0]
    # Update the initial prob  
    initExpCounts = np.zeros((nState,))  
    for t in range(len(sent)):
      initExpCounts += forwardProbs[t] * backwardProbs[t]
    
    return initExpCounts

  def updateTransitionCounts(self, forwardProbs, backwardProbs, sent):
    nState = self.init.shape[0]
    transExpCounts = np.zeros((nState, nState))
    # Update the transition probs
    for t in range(len(sent)-1):
      obs_arr = np.array([self.obs[j][sent[t+1]] if sent[t+1] in self.obs[j] else 0 for j in range(nState)])
      transExpCounts += np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans  
      if DEBUG:
        print('forward prob, obs, backward prob: ', np.tile(forwardProbs[t], (nState, 1)), obs_arr, backwardProbs[t])
        print('product: ', np.tile(forwardProbs[t], (nState, 1)).T * (obs_arr * backwardProbs[t+1]) * self.trans)
    
    if DEBUG:
      print('transExpCounts: ', transExpCounts)

    return transExpCounts
     
  def updateObservationCounts(self, forwardProbs, backwardProbs, sent):
    assert np.sum(forwardProbs, axis=1).all() and np.sum(backwardProbs, axis=1).all()
    # Update observation probs
    nState = len(self.init)
    newObsProbs = [{} for i in range(nState)]
    statePosteriors = ((forwardProbs * backwardProbs).T / np.sum(forwardProbs * backwardProbs, axis=1)).T
    if DEBUG:
      print('statePosteriors: ', statePosteriors)
    
    for t, w in enumerate(sent): 
      for i in range(nState):
        if w not in newObsProbs[i]:
          if DEBUG:
            print('New word added to the state')
          newObsProbs[i][w] = 0.
        newObsProbs[i][w] += statePosteriors[t][i]
    
    return newObsProbs

  def computeAvgLogLikelihood(self):
    ll = 0.
    for sent in self.trainCorpus:
      forwardProb = np.sum(self.forward(sent)[-1])
      ll += math.log(forwardProb)
    return ll / len(self.trainCorpus)

  def trainUsingEM(self, numIterations=10, writeModel=False, convergenceEpsilon=0.01):
    nState = self.init.shape[0]
   
    if writeModel:
      self.printModel('initial_model.txt')
 
    initCounts = np.zeros((nState,))
    transCounts = np.zeros((nState, nState))
    obsCounts = [{w: 0. for w in self.obs[i]} for i in range(nState)]  
    for epoch in range(numIterations): 
      AvgLogProb = self.computeAvgLogLikelihood()
      print('Epoch', epoch, 'Average Log Likelihood:', self.computeAvgLogLikelihood())  
      
      for sent in self.trainCorpus:
        forwardProbs = self.forward(sent)
        backwardProbs = self.backward(sent) 
        initCounts += self.updateInitialCounts(forwardProbs, backwardProbs, sent)
        transCounts += self.updateTransitionCounts(forwardProbs, backwardProbs, sent)
        obsCountsInc = self.updateObservationCounts(forwardProbs, backwardProbs, sent)
        for i in range(nState):
          for w in obsCountsInc[i]:
            if w not in obsCounts[i]:
              obsCounts[i][w] = obsCountsInc[i][w]
            else:
              obsCounts[i][w] += obsCountsInc[i][w]

      # Normalize
      self.init = initCounts / np.sum(initCounts) 
      
      totCounts = np.sum(transCounts, axis=1)
      for s in range(nState):
        if totCounts[s] == 0:
          # Not updating the transition arc if it is not used          
          self.trans[s] = self.trans[s]
        else:
          self.trans[s] = transCounts[s] / totCounts[s]
      
      for i in range(nState):
        normFactor = sum(obsCounts[i].values())
        if normFactor == 0:
          if DEBUG:
            print('norm factor for the obs is 0: potential bug')
          self.obs[i][w] = self.obs[i][w]
        for w in obsCounts[i]:
          self.obs[i][w] = obsCounts[i][w] / normFactor

      if writeModel:
        self.printModel(self.modelName + '_iter='+str(epoch)+'.txt')
    
  def viterbiDecode(self, sent, unkProb=10e-12):
    nState = self.init.shape[0]
    T = len(sent)
    scores = np.zeros((nState,))
    backPointers = np.zeros((T, nState), dtype=int)
    for i in range(nState):
      scores[i] = self.init[i] * self.obs[i][sent[0]] 

    for t, w in enumerate(sent[1:]):
      obs_arr = np.array([self.obs[i][w] if w in self.obs[i] else unkProb for i in range(nState)])
      candidates = np.tile(scores, (nState, 1)).T * self.trans * obs_arr
      backPointers[t+1] = np.argmax(candidates, axis=0)
      scores = np.max(candidates, axis=0)
      if DEBUG:
        print(scores)
    curState = np.argmax(scores)
    bestPath = [curState]
    for t in range(len(sent)-1, 0, -1):
      if DEBUG:
        print('curState: ', curState)
      curState = backPointers[t, curState]
      bestPath.append(curState)
    return bestPath[::-1], np.max(scores)
      
  def printModel(self, filename):
    nState = self.init.shape[0]
    initFile = open(filename+'_initialprobs.txt', 'w')
    for i in range(nState):
      initFile.write('%d\t%f\n' % (i, self.init[i]))
    initFile.close()

    transFile = open(filename+'_transitionprobs.txt', 'w')
    for i in range(nState):
      for j in range(nState):
        transFile.write('%d\t%d\t%f\n' % (i, j, self.trans[i][j]))
    transFile.close()

    obsFile = open(filename+'_observationprobs.txt', 'w')
    for i in range(nState):
      for w in self.obs[i]:
        obsFile.write('%d\t%s\t%f\n' % (i, w, self.obs[i][w]))
    obsFile.close()
  
#def printAlignment()
# for i in range(nState):

#def beamSearchDecode(self, sent):

if __name__ == '__main__':
  '''corpus = 'test.txt'
  f = open(corpus, 'r')
  a = f.read().strip().split('\n')
  corpusA = [w.split() for w in a[:10]]
  corpusB = [w.split() for w in a[10:]]
  print(corpusA)
  '''
  corpusA = [['a', 'b']]
  hmmA = DiscreteHMM(2, corpusA, modelName='A')
  hmmA.trainUsingEM(10, writeModel=True)
  
  '''hmmB = DiscreteHMM(2, corpusB, modelName='B')
  hmmB.trainUsingEM(10, writeModel=True)
  '''
  sent = ['a', 'b', 'b']
  '''
  pA = hmmA.forward(sent)
  pB = hmmB.forward(sent)
  print('pA, pB: ', pA, pB)'''
  print(hmmA.viterbiDecode(sent))
