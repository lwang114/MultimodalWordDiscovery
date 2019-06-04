import math
import numpy as np
import json
from nltk.tokenize import word_tokenize
from copy import deepcopy
import time
from scipy.stats import multivariate_normal

# Constant for NULL word at position zero in target sentence
NULL = "NULL"
# Minimum translation probability
PMIN = 10e-12
EPS = np.finfo(float).eps
DEBUG = False

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
def gaussian(x, mean, cov, cov_type='diag'):
  d = mean.shape[0]
  if cov_type == 'diag':
    assert np.min(np.diag(cov)) > 0.
    #log_norm_const = float(d) / 2. * np.log(2. * math.pi) + np.sum(np.log(np.diag(cov))) / 2.
    norm_const = np.sqrt(2. * math.pi) ** float(d)
    norm_const *= np.prod(np.sqrt(np.diag(cov))) 
    #log_prob = - log_norm_const - np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), x - mean) / 2. 
    #prob = np.exp(-np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), x - mean) / 2.) / norm_const
    x_z = x - mean
    prob = np.exp(- np.sum(x_z ** 2 / (2. * np.diag(cov)), axis=-1)) / norm_const 
  else:
    assert np.linalg.det(cov) > 0.
    chol_cov = np.linalg.cholesky(cov)
    #log_norm_const = float(d) / 2. * np.log(2. * math.pi) + np.log(np.linalg.det(chol_cov))
    #log_prob = - log_norm_const - np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), x - mean) / 2. 
    norm_const = np.sqrt(2. * math.pi) ** float(d)
    norm_const *= np.linalg.det(chol_cov)
    prob = np.exp(-np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), x - mean) / 2.) / norm_const

  #return log_prob 
  return prob
  #1 / (2 * math.pi * np.linalg.det(cov)) ** (d / 2) * math.exp(np.dot(np.dot(-(x - mean).T, np.linalg.inv(cov)), x - mean) / 2) 

def gmmProb(x, priors, means, covs):
  #log_prob = 0.
  m = priors.shape[0]
  if len(x.shape) == 1:
    probs = np.zeros((m,))
  elif len(x.shape) == 2:
    probs = np.zeros((m, x.shape[1]))
  else:
    raise ValueError('x has to be 1-d or 2-d array')

  #log_probs = np.zeros((m,))
  for i in range(m):
    if len(covs.shape) == 2:
      #log_probs[i] = gaussian(x, means[i], np.diag(covs[i]))
      probs[i] = gaussian(x, means[i], np.diag(covs[i]))
      #probs[i] = multivariate_normal.pdf(x, means[i], np.diag(covs[i]))
    else:
      #log_probs[i] = gaussian(x, means[i], covs[i])
      probs[i] = gaussian(x, means[i], covs[i])
  #return logDot(priors, log_probs)
  return np.dot(priors, probs)

class GMMWordDiscoverer:

    def __init__(self, sourceCorpusFile, targetCorpusFile, numMixtures):
        self.numMixtures = numMixtures
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences
        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences

        self.transMeans = {}                     # transMeans[e_i] is initialized with means of frames of sentence in which e_i appears (TODO: better initialization) 
        self.transVars = {}                      # transVars[e_i] is initialized similarly as transMeans; assume diagonal covariance (TODO: general covariance) 
        self.alignProb = []                     # alignProb[i][k_e_k^s][m] is a list of probabilities containing expected counts for each sentence
        self.lenProb = {}
        self.avgLogTransProb = float('-inf')
  
        # Read the corpus
        self.initialize(sourceCorpusFile, targetCorpusFile);
        self.fCorpus = self.fCorpus
        self.tCorpus = self.tCorpus
        # Initialize any additional data structures here (e.g. for probability model)

    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
    def initialize(self, fFileName, tFileName):
        fp = open(tFileName, 'r')
        tCorpus = fp.read().split('\n')
        self.tCorpus = [[NULL] + tw.split() for tw in tCorpus]
        fp.close()
        
        fCorpus = np.load(fFileName)
        
        self.fCorpus = [fCorpus[k] for k in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
        self.data_ids = [feat_id.split('_')[-1] for feat_id in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
        self.featDim = self.fCorpus[0].shape[1]
        
        return
    
    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=10, mixturePriorFile=None, transMeanFile=None, transVarFile=None, writeModel=False, modelPrefix='', epsilon=1e-5, smoothing=None):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities(smoothing=smoothing)         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationDensities(mixturePriorFile=mixturePriorFile, transMeanFile=transMeanFile, transVarFile=transVarFile)        # <you need to implement initializeTranslationProbabilities()>
        #self.avgLogTransProb = self.averageTranslationProbability()
        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 # <you need to implement printModel(filename)>
        #for i in range(numIterations):
        i = 1
        done = False
        while not done:
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
            if self.checkConvergence(epsilon):
              done = True
            print('Check convergence takes  %0.5f s to finish' % (time.time() - begin_time))  
            print ("Average Log Translation Probability: ", self.avgLogTransProb)
            
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
    def initializeWordTranslationDensities(self, mixturePriorFile=None, transMeanFile=None, transVarFile=None):
        # Initialize the translation mean and variance
        self.transMeans = {}
        self.transVars = {}
        self.mixturePriors = {}
        self.conceptCounts = {}

        # Initialize mixture means and priors with simple cyclic assignment of centroids
        if mixturePriorFile and transMeanFile and transVarFile:
          with open(mixturePriorFile, 'r') as f:
            self.mixturePriors = json.load(f)
            self.mixturePriors = {tw:np.array(self.mixturePriors[tw]) for tw in self.mixturePriors}

          with open(transMeanFile, 'r') as f:
            self.transMeans = json.load(f)
            self.transMeans = {tw:np.array(self.transMeans[tw]) for tw in self.transMeans}
          
          with open(transVarFile, 'r') as f:
            self.transVars = json.load(f)
            self.transVars = {tw:np.array(self.transVars[tw]) for tw in self.transVars}

        else:
          # TODO: better initialization such as KMeans++
          for ts, fs in zip(self.tCorpus, self.fCorpus):
            for tw in ts:
              if tw not in self.transMeans:
                self.transMeans[tw] = np.zeros((self.numMixtures, self.featDim))
                self.transVars[tw] = np.zeros((self.numMixtures, self.featDim))
                self.mixturePriors[tw] = np.ones((self.numMixtures,)) / self.numMixtures
                self.conceptCounts[tw] = np.zeros((self.numMixtures,))

              if fs.shape[0] == self.featDim:
                nframes = fs.shape[1]
              else:
                nframes = fs.shape[0]

              for kframe in range(nframes):
                m = kframe % self.numMixtures
                self.transMeans[tw][m] += fs[kframe]
                self.conceptCounts[tw][m] += 1 
        
          for tw in self.transMeans:
            for m in range(self.numMixtures):
              self.transMeans[tw][m] /= self.conceptCounts[tw][m]
              #self.transMeans[tw][i] += np.max(np.abs(self.transMeans[tw][i])) * np.random.normal(size=(self.featDim,))
          
          # Initialize mixture variance  
          for ts, fs in zip(self.tCorpus, self.fCorpus):
            for tw in ts:
              fLen = fs.shape[0]
              for kf in range(fLen):
                m = kf % self.numMixtures
                self.transVars[tw][m] += (fs[kf] - self.transMeans[tw][m]) ** 2
                 
          for tw in self.transMeans:
            for m in range(self.numMixtures):
              self.transVars[tw][m] /= self.conceptCounts[tw][m]
        
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
          normFactor = np.zeros((tLen, fLen))           
          for k_t, tw in enumerate(ts):    
            if tw not in align.keys():
              align[str(k_t)+'_'+tw] = np.zeros((self.numMixtures, fLen))
            
            tKey = str(k_t)+'_'+tw
            #for k_f in range(fLen):
            for m in range(self.numMixtures):
              #align[tKey][m, k_f] = np.log(self.mixturePriors[tw][m]) + gaussian(fs[k_f], self.transMeans[tw][m], np.diag(self.transVars[tw][m]))
              align[tKey][m] = self.mixturePriors[tw][m] * gaussian(fs, self.transMeans[tw][m], np.diag(self.transVars[tw][m]))
              #align[tKey][m] = self.mixturePriors[tw][m] * multivariate_normal.pdf(fs, self.transMeans[tw][m], np.diag(self.transVars[tw][m]))
    
            normFactor[k_t] = np.sum(align[tKey], axis=0) 
            #normFactor.append(logSum(align[tKey]))
            
          #normFactor = logSum(np.stack(normFactor, axis=1).T)
          normFactor = np.sum(normFactor, axis=0)

          for k_t, tw in enumerate(ts):
            tKey = str(k_t)+'_'+tw
            #align[tKey] = np.exp(align[tKey] - normFactor)
            align[tKey] = align[tKey] / normFactor

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
        normFactor = {} 
        for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
          for k_t, tw in enumerate(tSen):
            if tw not in self.transMeans.keys():
              self.transMeans[tw] = np.zeros((self.numMixtures, self.featDim))
              self.transVars[tw] = np.zeros((self.numMixtures, self.featDim))
              self.mixturePriors[tw] = np.zeros((self.numMixtures,)) / self.numMixtures
              normFactor[tw] = np.zeros((self.numMixtures,))
            
            fLen = fSen.shape[0]
            
            tKey = str(k_t)+'_'+tw
            for m in range(self.numMixtures):
              self.transMeans[tw][m] += np.dot(self.alignProb[i][tKey][m], fSen) 
              normFactor[tw][m] += np.sum(self.alignProb[i][tKey][m])
        
        # Normalize the estimated means over all audio frames
        for tw in self.transMeans.keys():
          for m in range(self.numMixtures):
            self.transMeans[tw][m] /= normFactor[tw][m] 

        # Update translation variance
        for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
          for k_t, tw in enumerate(tSen):
            tKey = str(k_t)+'_'+tw
            for m in range(self.numMixtures):
              #self.transVars[tw][m] += np.dot(self.alignProb[i][tKey][m], ((fSen - self.transMeans[tw][m]) ** 2)) 
              self.transVars[tw][m] += np.dot(self.alignProb[i][tKey][m], ((fSen - self.transMeans[tw][m]) ** 2))
        
        # Normalization over all the possible translation of the target word
        for tw in self.transMeans.keys():
          for m in range(self.numMixtures):
            if np.min(self.transVars[tw][m]) <= 0:
              print('zero variance norm factor: ', normFactor[tw][m])
            self.mixturePriors[tw][m] = normFactor[tw][m] / np.sum(normFactor[tw]) 
            self.transVars[tw][m] /= normFactor[tw][m]  
            if np.min(self.transVars[tw][m]) <= 0:
              print('after-norm zero variance norm factor: ', normFactor[tw][m])
  
        #pass
    
    # Compute average log probabilities
    def baumWelchFunction(self):
      avgTransProb = 0.  
      for i, (fs, ts) in enumerate(zip(self.fCorpus, self.tCorpus)):
        #if DEBUG:
        #  print(ts, fs, len(ts), len(fs))
        fLen = fs.shape[0]
        avgTransProb += math.log(self.lenProb[len(ts)-1][fLen]) - fLen * math.log(len(ts))
        #for k_f in range(fLen):
        for k_t, tw in enumerate(ts):
          for m in range(self.numMixtures):
            tKey = str(k_t)+'_'+tw
            avgTransProb += 1. / len(self.fCorpus) * np.dot(self.alignProb[i][tKey][m], np.log(EPS + self.mixturePriors[tw][m] * gaussian(fs, self.transMeans[tw][m], np.diag(self.transVars[tw][m]))))
            
            #avgTransProb += np.dot(self.alignProb[i][tKey][m], np.log(self.mixturePriors[tw][m] * multivariate_normal.pdf(fs, self.transMeans[tw][m], np.diag(self.transVars[tw][m]))))
      return avgTransProb
      
    def checkConvergence(self, eps=1e-5):
      avgLogTransProb = self.baumWelchFunction()
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
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ##
        fLen = fSen.shape[0]
        alignment = [0]*fLen
        alignProbs = []
        for k_f in range(fLen):
          bestScore = float('-inf')
          for k_t, tw in enumerate(tSen):
            if tw not in self.mixturePriors:
              score = PMIN
            else:
              score = gmmProb(fSen[k_f], self.mixturePriors[tw], self.transMeans[tw], self.transVars[tw])
            if score > bestScore:
              alignment[k_f] = k_t
              bestScore = score
          alignProbs.append(bestScore)

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

    # Return p(f_j | e_i), the probability that English word e_i generates non-English word f_j
    # (Can either return log probability or regular probability)
    '''def getWordTranslationProbability(self, f_j, e_i):
        # Implement this method
        if e_i in self.transMeans.keys():
          if f_j in self.trans[e_i].keys():
            return self.trans[e_i][f_j]
          else:
            return float('-inf')
        else:
          return float('-inf')
    '''
    # Write this model's probability distributions to file
    def printModel(self, filename):
        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        translateMeanFile = open(filename+'_translation_means.json', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        translateVarFile = open(filename+'_translation_variances.json', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        mixturePriorFile = open(filename+'_mixture_priors.json', 'w')

        # TODO: Make this more memory-efficient
        transMeans = {tw: self.transMeans[tw].tolist() for tw in sorted(self.transMeans.keys())}
        transVars = {tw: self.transVars[tw].tolist() for tw in sorted(self.transVars.keys())}
        mixPriors = {tw: self.mixturePriors[tw].tolist() for tw in sorted(self.mixturePriors.keys())}
         
        json.dump(transMeans, translateMeanFile, indent=4, sort_keys=True)
        json.dump(transVars, translateVarFile, indent=4, sort_keys=True)
        json.dump(mixPriors, mixturePriorFile, indent=4, sort_keys=True)
        for tLen in self.lenProb.keys():
          for fLen in self.lenProb[tLen].keys():
            lengthFile.write('{}\t{}\t{}\n'.format(tLen, fLen, self.lenProb[tLen][fLen]))
                
        lengthFile.close()
        translateMeanFile.close()
        translateVarFile.close()
        mixturePriorFile.close()

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
    '''datapath = '/home/lwang114/data/flickr/flickr_40k_speech_mbn/'
    featfile = 'flickr_40k_speech_train.npz'
    feats = np.load(datapath + featfile)
    feat_keys = feats.keys()[0:6:5]
    print(feat_keys)
    np.savez('small.npz', *[feats[k] for k in feat_keys]) 
      
    print(np.exp(logSum([math.log(3), math.log(2)])))
    print(np.exp(logDot(np.array([math.log(3)]), np.array([math.log(2)]))))
    print(np.exp(logDot(np.array([[math.log(3), math.log(2)], [math.log(5), math.log(1)]]), np.array([[math.log(2), math.log(6)], [math.log(4), math.log(7)]]))))
    
    model = GMMWordDiscoverer('small.npz', 'small.txt', 3)
    model.trainUsingEM(writeModel=False)
    '''
    datapath = '../data/flickr30k/audio_level/'
    tr_src_file = 'flickr_bnf_train_src.npz'
    tr_trg_file = 'flickr_bnf_train_trg.txt'
    tx_src_file = 'flickr_bnf_test_src.npz'
    tx_trg_file = 'flickr_bnf_test_trg.txt'
    src_file = 'flickr_bnf_all_src.npz'
    trg_file = 'flickr_bnf_all_trg.txt'

    model = GMMWordDiscoverer(datapath+src_file, datapath+trg_file, 10)
    model.trainUsingEM(writeModel=True)
    
    #model.trainUsingEM(writeModel=True, mixturePriorFile='model_iter=2.txt_mixture_priors.json', transMeanFile='model_iter=2.txt_translation_means.json', transVarFile='model_iter=2.txt_translation_variances.json')
    #model.initializeWordTranslationDensities(mixturePriorFile='model_iter=2.txt_mixture_priors.json', transMeanFile='model_iter=2.txt_translation_means.json', transVarFile='model_iter=2.txt_translation_variances.json')
    model.printAlignment('flickr30k_pred_alignment')
     
    #model.trainUsingEM(transProbFile='models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt', writeModel=True)
    #model.initializeWordTranslationProbabilities(transProbFile='models/mar18_flickr30k_phoneme_level_ibm2/model_iter=7.txt_translationprobs.txt')
    #model.initializeAlignmentProbabilities(alignProbFile='models/mar18_flickr30k_phoneme_level_ibm2/model_iter=7.txt_alignpriors.txt')
    
