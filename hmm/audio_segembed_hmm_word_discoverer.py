import numpy as np
import math
import json
from audio_hmm_word_discoverer import *
from scipy.misc import logsumexp
import scipy.signal as signal
import scipy.interpolate as interpolate 

NULL = "NULL"
DEBUG = False
ORDER = 'C'

# TODO: Incorporate this under HMM class
# Audio-level word discovery model
# * The transition matrix is assumed to be Toeplitz 
class SegEmbedHMMWordDiscoverer:
  def __init__(self, numMixtures, frameDim, embedDim, 
              sourceCorpusFile, targetCorpusFile, 
              landmarkFile,
              modelDir=None,
              minWordLen=20,
              maxWordLen=100,
              modelName='audio_segembed_hmm_word_discoverer', maxLen=2000):
    self.modelName = modelName 
    self.initProbFile = None
    self.transProbFile = None
    self.obsModelFile = None
    if modelDir:
      self.initProbFile = modelDir + "model_final_initialprobs.txt" 
      self.transProbFile = modelDir + "model_final_transitionprobs.txt"
      self.obsModelFile = modelDir + "model_final_obs_model"
    
    self.init = {}
    self.trans = {}                 # trans[l][i][j] is the probabilities that target word e_j is aligned after e_i is aligned in a target sentence e of length l  
    self.lenProb = {}
    self.assignments = []
    self.segmentations = []
    self.embeddings = []
    self.numMixtures = numMixtures
    self.avgLogTransProb = float('-inf')
    self.embedDim = embedDim
    self.frameDim = frameDim

    # Initialize data structures for storing training data
    self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

    self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences
    
    # Read the corpus
    self.initialize(landmarkFile, sourceCorpusFile, targetCorpusFile, maxLen=maxLen);
        
  def initialize(self, landmarkFile, fFileName, tFileName, initProbFile=None, transProbFile=None, obsModelFile=None, initMethod="rand", fixedVariance=0.02, maxLen=2000):
    fp = open(tFileName)
    tCorpus = fp.read().split('\n')
 
    # XXX XXX    
    self.tCorpus = [[NULL] + tw.split() for tw in tCorpus]
    fp.close()
        
    fCorpus = np.load(fFileName) 
    # XXX XXX
    self.fCorpus = [fCorpus[k] for k in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
    self.fCorpus = [fSen[:maxLen] for fSen in self.fCorpus] 
    self.featDim = self.fCorpus[0].shape[1]

    self.data_ids = [feat_id.split('_')[-1] for feat_id in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
     
    landmarks = np.load(landmarkFile)
    #XXX XXX
    for lm_id in sorted(landmarks, key=lambda x:int(x.split('_')[-1])):
      segmentation = []
      for b in landmarks[lm_id]:
        if b <= maxLen:
          segmentation.append(b)
        else:
          segmentation.append(maxLen)
          break
      self.segmentations.append(segmentation)
     
    for i, (fSen, segmentation) in enumerate(zip(self.fCorpus, self.segmentations)):      
      self.embeddings.append(self.getSentEmbeds(fSen, segmentation, frameDim=self.frameDim))
      #if DEBUG:
      if i >= 1 and i <= 4: 
        print(i, self.embeddings[i])

    self.acoustic_model = AudioHMMWordDiscoverer(self.numMixtures, self.frameDim, 
                        fCorpus=self.embeddings, tCorpus=self.tCorpus, 
                        initProbFile=initProbFile,
                        transProbFile=transProbFile,
                        obsModelFile=obsModelFile,
                        initMethod=initMethod,
                        maxLen=maxLen, fixedVariance=fixedVariance)
    print("Finish initialization of acoustic model")
    
  def trainUsingEM(self, numIterations=30, numAMSteps=1, modelPrefix='', writeModel=False):
    if writeModel:
      self.acoustic_model.printModel('initial_model.txt')
    
    for epoch in range(numIterations): 
      #AvgLogProb = self.computeAvgLogLikelihood()
      print("Start training iteration "+str(epoch))
      begin_time = time.time()

      self.acoustic_model.trainUsingEM(numIterations=numAMSteps)
      print("Acoustic model training takes %0.5f s to finish" % (time.time()-begin_time))
      
      if writeModel:
        self.acoustic_model.printModel(modelPrefix+"model_iter="+str(epoch))

    if writeModel:
      self.acoustic_model.printModel(modelPrefix+'model_final')

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
        y_new = f(x_new).flatten(ORDER) #.flatten("F")
    elif technique == "resample":
        y_new = signal.resample(y.astype("float32"), n, axis=1).flatten(ORDER) #.flatten("F")
    elif technique == "rasanen":
        # Taken from Rasenen et al., Interspeech, 2015
        n_frames_in_multiple = int(np.floor(y.shape[1] / n)) * n
        y_new = np.mean(
            y[:, :n_frames_in_multiple].reshape((d_frame, n, -1)), axis=-1
            ).flatten(ORDER) #.flatten("F")
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
     
  #def getSentAssignScores(self, embeddings, tSen):
  #  return [self.getAssignScores(embedding, tSen) for _, embedding in enumerate(embeddings)]

  def assign(self, i):
    embeddings = self.embeddings[i]
    segmentation = self.segmentations[i]
    return self.acoustic_model.align(embeddings, self.tCorpus[i])
     
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
    print(len(assignment), len(durations), len(assign_scores))
    print(segmentation[-1], np.sum(durations))

    for j, scores, dur in zip(assignment, assign_scores, durations):  
      alignment.extend([j] * int(dur))
      align_probs.extend([scores] * int(dur))
    
    return alignment, align_probs
    
  def printAlignment(self, filePrefix, isPhoneme=True):
    f = open(filePrefix+'.txt', 'w')
    aligns = []
    if DEBUG:
      print(len(self.fCorpus))
    for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
      alignment, alignProbs = self.align(i)
      print(i)
      #if DEBUG:
      #  print(fSen, tSen)
      #  print(type(alignment[1]))
      align_info = {
            'index': self.data_ids[i],
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
  #trainingCorpusFile = 'test_translation.txt' 
  #'../data/flickr30k/phoneme_level/flickr30k.txt'
  initProbFile = "exp/"#'models/apr18_tiny_hmm_translate/A_iter=9.txt_initialprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_initialprobs.txt'
  #transProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_transitionprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_transitionprobs.txt'
  #obsProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_observationprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_observationprobs.txt'
  sourceCorpusFile = "../data/flickr30k/audio_level/flickr_bnf_all_src.npz" #"../data/flickr30k/audio_level/flickr_mfcc_cmvn_htk.npz" 
  targetCorpusFile = "../data/flickr30k/audio_level/flickr_bnf_all_trg.txt"
  landmarkFile = "../data/flickr30k/audio_level/flickr30k_gold_landmarks_mbn.npz" #"../data/flickr30k/audio_level/flickr_landmarks_mbn_combined.npz" #"../data/flickr30k/audio_level/flickr30k_gold_landmarks_mbn.npz"
  model = SegEmbedHMMWordDiscoverer(5, 12, 120, sourceCorpusFile, targetCorpusFile, landmarkFile=landmarkFile, modelName='test_segembed_hmm', maxLen=2000)
  #model = HMMWordDiscoverer(trainingCorpusFile, initProbFile, transProbFile, obsProbFile, modelName='A')
  model.trainUsingEM(10, writeModel=True)
  model.printAlignment('alignment')
  with open("alignment.json", "r") as f:
    a = json.load(f)
  print("alignment[0].shape: ", np.array(a[0]["alignment"]).shape)
