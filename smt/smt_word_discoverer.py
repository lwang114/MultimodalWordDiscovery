import math
import numpy as np
import json
from nltk.tokenize import word_tokenize
from copy import deepcopy
# Constant for NULL word at position zero in target sentence
NULL = "NULL"
# Minimum translation probability
PMIN = 10e-12
DEBUG = False
# Your task is to finish implementing IBM Model 1 in this class
class IBMModel1:

    def __init__(self, trainingCorpusFile):
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences

        self.trans = {}                     # trans[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
        self.alignProb = []                     # alignProb[i][j_f_j^s][k_e_k^s] is a list of probabilities containing expected counts for each sentence
        self.lenProb = {}
        self.avgLogTransProb = float('-inf')
  
        # Read the corpus
        self.initialize(trainingCorpusFile);
        self.fCorpus = self.fCorpus
        self.tCorpus = self.tCorpus
        # Initialize any additional data structures here (e.g. for probability model)

    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
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
                for tw in tTokenized:
                    if tw not in self.trans:
                        self.trans[tw] = {};
                    for fw in fTokenized:
                        if fw not in self.trans[tw]:
                             self.trans[tw][fw] = 1 / len(fTokenized)
                        else:
                            self.trans[tw][fw] =  self.trans[tw][fw] + 1 / len(fTokenized)
            else:
                i = -1
                j += 1
            i +=1
        f.close()
        return

    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=10, transProbFile=None, writeModel=False, epsilon=1e-5, smoothing=None):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities(smoothing=smoothing)         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationProbabilities(transProbFile=transProbFile)        # <you need to implement initializeTranslationProbabilities()>
        #self.avgLogTransProb = self.averageTranslationProbability()
        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 # <you need to implement printModel(filename)>
        #for i in range(numIterations):
        i = 1
        while not self.checkConvergence(epsilon):
            print ("Starting training iteration "+str(i))
            print ("Average Log Translation Probability: ", self.avgLogTransProb)
            # Run E-step: calculate expected counts using current set of parameters
            self.computeExpectedCounts()                     # <you need to implement computeExpectedCounts()>
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationProbabilities()            # <you need to implement updateTranslationProbabilities()>
            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel('model_iter='+str(i)+'.txt')     # <you need to implement printModel(filename)>
            i += 1

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
            self.lenProb[len(ts)-1][len(fs)] = 1
          else:
            self.lenProb[len(ts)-1][len(fs)] += 1
        
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
    def initializeWordTranslationProbabilities(self, transProbFile=None):
        # Implement this method
        self.trans = {}
        if transProbFile:
          f = open(transProbFile)
          for line in f:
            tw, fw, prob = line.strip().split()
            if tw not in self.trans.keys():
              self.trans[tw] = {}
            self.trans[tw][fw] = float(prob)
             
          f.close()
        else:
          for ts, fs in zip(self.tCorpus, self.fCorpus):
            for tw in ts:
              for fw in fs:
                if tw not in self.trans.keys():
                  self.trans[tw] = {}  
                if fw not in self.trans[tw].keys():
                  self.trans[tw][fw] = 1
        
          for tw in self.trans:
            totCount = sum(self.trans[tw].values())
            for fw in self.trans[tw].keys():
              #if DEBUG:
              #  if self.trans[tw][fw] > 1:
              #    print(self.trans[tw][fw])
              self.trans[tw][fw] = self.trans[tw][fw] / totCount 
      
    # Run E-step: calculate expected counts using current set of parameters
    def computeExpectedCounts(self):
        # Implement this method
        # Reset align every iteration
        self.alignProb = []
        for i, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
          align = {}
          for j, fw in enumerate(fs):
            if fw not in align.keys():
              align[str(j)+'_'+fw] = {}
            for k, tw in enumerate(ts):
              align[str(j)+'_'+fw][str(k)+'_'+tw] = self.trans[tw][fw]
          
          # Normalization across target words in the sentence
          for j, fw in enumerate(fs):
            fKey = str(j)+'_'+fw
            totCount = sum(align[fKey].values())
            for k, tw in enumerate(ts):
              tKey = str(k)+'_'+tw
              align[fKey][tKey] /= totCount
              if DEBUG and align[fKey][tKey] == 0:
                 print(j, k, fw, tw, ts, totCount)
          self.alignProb.append(align)
        # Update the expected counts
        #pass

    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationProbabilities(self):
        # Implement this method
        n_sentence = min(len(self.tCorpus), len(self.fCorpus))
        self.trans = {}
        # Sum all the expected counts across sentence for pairs of translation
        for i in range(n_sentence):
          for j, tw in enumerate(self.tCorpus[i]):
            if tw not in self.trans.keys():
              self.trans[tw] = {}
            for k, fw in  enumerate(self.fCorpus[i]): 
              fKey = str(k)+'_'+fw
              tKey = str(j)+'_'+tw
              if fw not in self.trans[tw].keys():
                self.trans[tw][fw] = self.alignProb[i][fKey][tKey]  
              else:
                self.trans[tw][fw] += self.alignProb[i][fKey][tKey]   

        # Normalization over all the possible translation of the target word
        for tw in self.trans.keys():
          totCount = sum(self.trans[tw].values())
          for fw in self.trans[tw]:
            #if DEBUG:
            #  print(tw, totCount, self.trans[tw][fw])          
            self.trans[tw][fw] = self.trans[tw][fw] / totCount    
        #pass
    
    # Compute average log probabilities
    def averageTranslationProbability(self):
      avgTransProb = 0.  
      if DEBUG:
        print(len(self.fCorpus), self.fCorpus[0])
      for fs, ts in zip(self.fCorpus, self.tCorpus):
        #if DEBUG:
        #  print(ts, fs, len(ts), len(fs))
           
        avgTransProbs += math.log(self.lenProb[len(ts)-1][len(fs)]) - len(fs) * math.log(len(ts))
        for fw in fs:   
          avgTransWord = 0.
          for tw in ts:
            avgTransWord += self.trans[tw][fw]
          avgTransProb += math.log(avgTransWord)
      return avgTransProb / len(self.fCorpus)
      
    def checkConvergence(self, eps=1e-5):
      avgLogTransProb = self.averageTranslationProbability()
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
        alignment = [0]*len(fSen)
        alignProbs = []
        for i, fw in enumerate(fSen):
          bestProb = float('-inf')
          alignProb = []
          for j, tw in enumerate(tSen):
            alignProb.append(self.trans[tw][fw])
            if self.trans[tw][fw] > bestProb:
              alignment[i] = j
              bestProb = self.trans[tw][fw] 
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

    # Return p(f_j | e_i), the probability that English word e_i generates non-English word f_j
    # (Can either return log probability or regular probability)
    def getWordTranslationProbability(self, f_j, e_i):
        # Implement this method
        if e_i in self.trans.keys():
          if f_j in self.trans[e_i].keys():
            return self.trans[e_i][f_j]
          else:
            return float('-inf')
        else:
          return float('-inf')
    
    # Write this model's probability distributions to file
    def printModel(self, filename):
        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        translateProbFile = open(filename+'_translationprobs.txt', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        for tw in sorted(self.trans.keys()):
          for fw in sorted(self.trans[tw].keys()): 
            translateProbFile.write('{}\t{}\t{}\n'.format(tw, fw, self.trans[tw][fw]))
        
        for tLen in self.lenProb.keys():
          for fLen in self.lenProb[tLen].keys():
            lengthFile.write('{}\t{}\t{}\n'.format(tLen, fLen, self.lenProb[tLen][fLen]))
        
        lengthFile.close();
        translateProbFile.close()

    # Write the predicted alignment to file
    def printAlignment(self, file_prefix, is_phoneme=True):
      f = open(file_prefix+'.txt', 'w')
      aligns = []
      if DEBUG:
        print(len(self.fCorpus))
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment, alignProbs = self.align(fSen, tSen)
        #if DEBUG:
        #  print(fSen, tSen)
        #  print(alignment)
        align_info = {
            'index': i,
            'image_concepts': tSen, 
            'caption': fSen,
            'alignment': alignment,
            'align_probs': alignProbs,
            'is_phoneme': is_phoneme
          }
        aligns.append(align_info)
        f.write('%s\n%s\n' % (tSen, fSen))
        for a in alignment:
          f.write('%d ' % a)
        f.write('\n\n')

      f.close()
    
      # Write to a .json file for evaluation
      with open(file_prefix+'.json', 'w') as f:
        json.dump(aligns, f, indent=4, sort_keys=True)            


class IBMModel2:
    def __init__(self, trainingCorpusFile):
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences

        self.trans = {}                     # trans[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
        self.alignProb = []                 # alignProb[i][j_f_j^s][k_e_k^s] is a list of probabilities containing expected counts for each sentence
        self.alignPriors = {}               # alignPriors[tLen][fLen][i][j] is the probabilities of alignment given the position of the target and foreign words 
      
        self.lenProb = {}
        self.avgLogTransProb = float('-inf')
  
        # Read the corpus
        self.initialize(trainingCorpusFile);
        self.fCorpus = self.fCorpus
        self.tCorpus = self.tCorpus
        
    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
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
                for tw in tTokenized:
                    if tw not in self.trans:
                        self.trans[tw] = {};
                    for fw in fTokenized:
                        if fw not in self.trans[tw]:
                             self.trans[tw][fw] = 1 / len(fTokenized)
                        else:
                            self.trans[tw][fw] =  self.trans[tw][fw] + 1 / len(fTokenized)
            else:
                i = -1
                j += 1
            i +=1
        f.close()
        return

    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=50, writeModel=False, transProbFile=None, alignPriorFile=None, epsilon=1e-2):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities()         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationProbabilities(transProbFile)        # <you need to implement initializeTranslationProbabilities()>
        self.initializeAlignmentProbabilities(alignPriorFile)
        
        print("Initial log likelihood: ", self.averageTranslationProbability())
        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 # <you need to implement printModel(filename)>
        
        for i in range(numIterations):
            #i = 1
            #while not self.checkConvergence(epsilon):
            print ("Starting training iteration "+str(i))
            print ("Average Log Translation Probability: ", self.avgLogTransProb)
            # Run E-step: calculate expected counts using current set of parameters
            self.computeExpectedCounts()                     # <you need to implement computeExpectedCounts()>
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationProbabilities()            # <you need to implement updateTranslationProbabilities()>
            self.updateAlignmentProbabilities()
 
            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel('model_iter='+str(i)+'.txt')     # <you need to implement printModel(filename)>
            #i += 1

    # Compute translation length probabilities q(m|n)
    def computeTranslationLengthProbabilities(self):
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
            self.lenProb[len(ts)-1][len(fs)] = 1
          else:
            self.lenProb[len(ts)-1][len(fs)] += 1

        for tl in self.lenProb.keys():
          totCount = sum(self.lenProb[tl].values())  
          for fl in self.lenProb[tl].keys():
            self.lenProb[tl][fl] = self.lenProb[tl][fl] / totCount 

    # Set initial values for the translation probabilities p(f|e)
    def initializeWordTranslationProbabilities(self, transProbFile=None):
        # Implement this method
        self.trans = {}
        if transProbFile:
          f = open(transProbFile)
          for line in f:
            tw, fw, prob = line.strip().split()
            if tw not in self.trans.keys():
              self.trans[tw] = {}
            self.trans[tw][fw] = float(prob)
             
          f.close()
        else:
          for ts, fs in zip(self.tCorpus, self.fCorpus):
            for tw in ts:
              for fw in fs:
                if tw not in self.trans.keys():
                  self.trans[tw] = {}  
                if fw not in self.trans[tw].keys():
                  self.trans[tw][fw] = 1
        
          for tw in self.trans:
            totCount = sum(self.trans[tw].values())
            for fw in self.trans[tw].keys():
              #if DEBUG:
              #  if self.trans[tw][fw] > 1:
              #    print(self.trans[tw][fw])
              self.trans[tw][fw] = self.trans[tw][fw] / totCount 
      
    def initializeAlignmentProbabilities(self, alignProbFile=None):
      for k, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
        if DEBUG:
          if len(ts) == 1:
            print(k, fs, ts)
        
        if alignProbFile:
          f = open(alignProbFile)
          for line in f:
            fLen, tLen, j, i, prob = line.strip().split()
            fLen, tLen, j, i, prob = int(fLen), int(tLen), int(j), int(i), float(prob)
            if tLen not in self.alignPriors:
              self.alignPriors[tLen] = {}
              self.alignPriors[tLen][fLen] = np.zeros((tLen, fLen))
            elif fLen not in self.alignPriors[tLen]:
              self.alignPriors[tLen][fLen] = np.zeros((tLen, fLen))
            self.alignPriors[tLen][fLen][i][j] = prob
          return

        if not len(ts) in self.alignPriors: 
          self.alignPriors[len(ts)] = {}
          self.alignPriors[len(ts)][len(fs)] = 1 / len(ts) * np.ones((len(ts), len(fs)))
        elif not len(fs) in self.alignPriors[len(ts)]:
          self.alignPriors[len(ts)][len(fs)] = 1 / len(ts) * np.ones((len(ts), len(fs)))

    # Run E-step: calculate expected counts using current set of parameters
    def computeExpectedCounts(self):
        # Implement this method
        # Reset align every iteration
        self.alignProb = []
        for i, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
          align = {}
          for j, fw in enumerate(fs):
            if fw not in align.keys():
              align[str(j)+'_'+fw] = {}
            for k, tw in enumerate(ts):
              align[str(j)+'_'+fw][str(k)+'_'+tw] = self.alignPriors[len(ts)][len(fs)][k][j] * self.trans[tw][fw]
          
          # Normalization across target words in the sentence
          for j, fw in enumerate(fs):
            fKey = str(j)+'_'+fw
            totCount = sum(align[fKey].values())
            for k, tw in enumerate(ts):
              tKey = str(k)+'_'+tw
              if DEBUG and align[fKey][tKey] == 0:
                 print(j, k, fw, tw, ts, totCount)
              align[fKey][tKey] /= totCount
              
          self.alignProb.append(align)
        # Update the expected counts
        #pass

    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationProbabilities(self):
        # Implement this method
        n_sentence = min(len(self.tCorpus), len(self.fCorpus))
        self.trans = {}
        # Sum all the expected counts across sentence for pairs of translation
        for i in range(n_sentence):
          for j, tw in enumerate(self.tCorpus[i]):
            if tw not in self.trans.keys():
              self.trans[tw] = {}
            for k, fw in  enumerate(self.fCorpus[i]): 
              fKey = str(k)+'_'+fw
              tKey = str(j)+'_'+tw
              if fw not in self.trans[tw].keys():
                self.trans[tw][fw] = self.alignProb[i][fKey][tKey]  
              else:
                self.trans[tw][fw] += self.alignProb[i][fKey][tKey]   

        # Normalization over all the possible translation of the target word
        for tw in self.trans.keys():
          totCount = sum(self.trans[tw].values())
          for fw in self.trans[tw]:
            #if DEBUG:
            #  print(tw, totCount, self.trans[tw][fw])          
            self.trans[tw][fw] = self.trans[tw][fw] / totCount    
            # Threshold to avoid underflow
            if self.trans[tw][fw] < PMIN:
              self.trans[tw][fw] = PMIN
        #pass
 
    def updateAlignmentProbabilities(self):
      newAlignPriors = {tLen: {fLen: np.zeros((int(tLen), int(fLen))) for fLen in self.alignPriors[tLen]} for tLen in self.alignPriors}
      for fs, ts in zip(self.fCorpus, self.tCorpus):
        for i, tw in enumerate(ts):
          for j, fw in enumerate(fs):
            newAlignPriors[len(ts)][len(fs)][i][j] += self.alignPriors[len(ts)][len(fs)][i][j] * self.trans[tw][fw]    

      for tLen in self.alignPriors:
        for fLen in self.alignPriors[tLen]:
          self.alignPriors[tLen][fLen] = newAlignPriors[tLen][fLen] / np.sum(newAlignPriors[tLen][fLen], axis=0) 
 
    def averageTranslationProbability(self):
      avgTransProb = 0.
      if DEBUG:
        print(len(self.fCorpus), self.fCorpus[0])
      for fs, ts in zip(self.fCorpus, self.tCorpus):
        #if DEBUG:
        #  print(ts, fs, len(ts), len(fs))
        avgTransProb += math.log(self.lenProb[len(ts)-1][len(fs)])
        for j, fw in enumerate(fs):   
          avgTransWord = 0. 
          for i, tw in enumerate(ts):
            avgTransWord += self.alignPriors[len(ts)][len(fs)][i][j] * self.trans[tw][fw]
          avgTransProb += math.log(avgTransWord)
      return avgTransProb / len(self.fCorpus)
      
    def checkConvergence(self, eps=1e-5):
      avgLogTransProb = self.averageTranslationProbability()
      if self.avgLogTransProb == float('-inf'):
        self.avgLogTransProb = avgLogTransProb
        return 0
      if abs((self.avgLogTransProb - avgLogTransProb) / avgLogTransProb) < eps:
        self.avgLogTransProb = avgLogTransProb
        return 1
      
      self.avgLogTransProb = avgLogTransProb  
      return 0
    
    # TODO: Modify for IBM Model 2
    # Returns the best alignment between fSen and tSen, according to your model
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###
        alignment = [0]*len(fSen)
        alignProbs = []
        for j, fw in enumerate(fSen):
          bestProb = float('-inf')
          alignProb = []
          for i, tw in enumerate(tSen):
            alignProb.append(self.alignPriors[len(tSen)][len(fSen)][i][j] * self.trans[tw][fw])
            score = 0.
            if len(tSen) not in self.alignPriors:
              score = 1 / len(tSen) * self.trans[tw][fw]
            elif len(fSen) not in self.alignPriors[len(tSen)]:
              score = 1 / len(tSen) * self.trans[tw][fw]
            else:
              score = self.alignPriors[len(tSen)][len(fSen)][i][j] * self.trans[tw][fw]

            if score > bestProb:
              alignment[j] = i
              bestProb = score 
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

    # Return p(f_j | e_i), the probability that English word e_i generates non-English word f_j
    # (Can either return log probability or regular probability)
    def getWordTranslationProbability(self, f_j, e_i):
        # Implement this method
        if e_i in self.trans.keys():
          if f_j in self.trans[e_i].keys():
            return self.trans[e_i][f_j]
          else:
            return float('-inf')
        else:
          return float('-inf')
    
    # Write this model's probability distributions to file
    def printModel(self, filename):
        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        translateProbFile = open(filename+'_translationprobs.txt', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file 
        alignPriorFile = open(filename+'_alignpriors.txt', 'w')
        
        # Implement this method (make your output legible and informative)
        for tLen in sorted(list(self.alignPriors.keys())):
          for fLen in sorted(list(self.alignPriors[tLen].keys())):
            for j in range(fLen):
              for i in range(tLen):
                alignPriorFile.write('{}\t{}\t{}\t{}\t{}\n'.format(fLen, tLen, j, i, self.alignPriors[tLen][fLen][i][j]))

        for tw in sorted(self.trans.keys()):
          for fw in sorted(self.trans[tw].keys()): 
            translateProbFile.write('{}\t{}\t{}\n'.format(tw, fw, self.trans[tw][fw]))
        
        for tLen in self.lenProb.keys():
          for fLen in self.lenProb[tLen].keys():
            lengthFile.write('{}\t{}\t{}\n'.format(tLen, fLen, self.lenProb[tLen][fLen]))
        #alignProbFile.close()
        lengthFile.close();
        translateProbFile.close()

    # Write the predicted alignment to file
    def printAlignment(self, file_prefix, is_phoneme=True):
      f = open(file_prefix+'.txt', 'w')
      aligns = []
      if DEBUG:
        print(len(self.fCorpus))
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment, alignProbs = self.align(fSen, tSen)
        #if DEBUG:
        #  print(fSen, tSen)
        #  print(alignment)
        align_info = {
            'index': i,
            'image_concepts': tSen, 
            'caption': fSen,
            'alignment': alignment,
            'align_probs': alignProbs,
            'is_phoneme': is_phoneme
          }
        aligns.append(align_info)
        f.write('%s\n%s\n' % (tSen, fSen))
        for a in alignment:
          f.write('%d ' % a)
        f.write('\n\n')

      f.close()
    
      # Write to a .json file for evaluation
      with open(file_prefix+'.json', 'w') as f:
        json.dump(aligns, f, indent=4, sort_keys=True)            


class IBMModel2PH:
    def __init__(self, trainingCorpusFile):
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences

        self.trans = {}                     # trans[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
        self.alignProb = []                 # alignProb[i][j_f_j^s][k_e_k^s] is a list of probabilities containing expected counts for each sentence
        self.alignPriors = {}               # alignPriors[tLen][fLen][i][j] is the probabilities of alignment given the position of the target and foreign words 
      
        self.lenProb = {}
        self.avgLogTransProb = float('-inf')
  
        # Read the corpus
        self.initialize(trainingCorpusFile);
        self.fCorpus = self.fCorpus
        self.tCorpus = self.tCorpus
        
    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
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
                for tw in tTokenized:
                    if tw not in self.trans:
                        self.trans[tw] = {};
                    for fw in fTokenized:
                        if fw not in self.trans[tw]:
                             self.trans[tw][fw] = 1 / len(fTokenized)
                        else:
                            self.trans[tw][fw] =  self.trans[tw][fw] + 1 / len(fTokenized)
            else:
                i = -1
                j += 1
            i +=1
        f.close()
        return

    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=50, writeModel=False, transProbFile=None, alignPriorFile=None, epsilon=1e-2):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities()         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationProbabilities(transProbFile)        # <you need to implement initializeTranslationProbabilities()>
        self.initializeAlignmentProbabilities(alignPriorFile)
        
        print("Initial log likelihood: ", self.averageTranslationProbability())
        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 # <you need to implement printModel(filename)>
        
        for i in range(numIterations):
            #i = 1
            #while not self.checkConvergence(epsilon):
            print ("Starting training iteration "+str(i))
            print ("Average Log Translation Probability: ", self.avgLogTransProb)
            # Run E-step: calculate expected counts using current set of parameters
            self.computeExpectedCounts()                     # <you need to implement computeExpectedCounts()>
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationProbabilities()            # <you need to implement updateTranslationProbabilities()>
            self.updateAlignmentProbabilities()
 
            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel('model_iter='+str(i)+'.txt')     # <you need to implement printModel(filename)>
            #i += 1

    # Compute translation length probabilities q(m|n)
    def computeTranslationLengthProbabilities(self):
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
            self.lenProb[len(ts)-1][len(fs)] = 1
          else:
            self.lenProb[len(ts)-1][len(fs)] += 1

        for tl in self.lenProb.keys():
          totCount = sum(self.lenProb[tl].values())  
          for fl in self.lenProb[tl].keys():
            self.lenProb[tl][fl] = self.lenProb[tl][fl] / totCount 

    # Set initial values for the translation probabilities p(f|e)
    def initializeWordTranslationProbabilities(self, transProbFile=None):
        # Implement this method
        self.trans = {}
        if transProbFile:
          f = open(transProbFile)
          for line in f:
            tw, fw, prob = line.strip().split()
            if tw not in self.trans.keys():
              self.trans[tw] = {}
            self.trans[tw][fw] = float(prob)
             
          f.close()
        else:
          for ts, fs in zip(self.tCorpus, self.fCorpus):
            for tw in ts:
              for fw in fs:
                if tw not in self.trans.keys():
                  self.trans[tw] = {}  
                if fw not in self.trans[tw].keys():
                  self.trans[tw][fw] = 1
        
          for tw in self.trans:
            totCount = sum(self.trans[tw].values())
            for fw in self.trans[tw].keys():
              #if DEBUG:
              #  if self.trans[tw][fw] > 1:
              #    print(self.trans[tw][fw])
              self.trans[tw][fw] = self.trans[tw][fw] / totCount 
      
    def initializeAlignmentProbabilities(self, alignProbFile=None):
      for i, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
        if DEBUG:
          if len(ts) == 1:
            print(i, fs, ts)
        
        if alignProbFile:
          f = open(alignProbFile)
          for line in f:
            i_prev, i_cur, prob = line.strip().split()
            i_prev, i_cur, prob = int(i), int(j), float(prob)
            if i_prev not in self.alignPriors:
              self.alignPriors[i_prev] = {}
            self.alignPriors[i_prev][i_cur] = prob
          return

        for i_prev in range(len(ts)): 
          for i_cur in range(len(fs)):
            if i_prev not in self.alignPriors:
              self.alignPriors[i_prev] = {}
              self.alignPriors[i_prev][i_cur] = 1
      
      for i in self.alignPriors:
        totCounts = sum(self.alignPriors[i].values())    
        for j in self.alignPriors[i]:
          self.alignPriors[i][j] /= totCounts


    # Run E-step: calculate expected counts using current set of parameters
    def computeExpectedCounts(self):
        # Implement this method
        # Reset align every iteration
        self.expTransCounts = []
        self.expAlignCounts = []
        self.expInitCounts = []
        for i, (ts, fs) in enumerate(zip(self.tCorpus, self.fCorpus)):
          expTransCount = {}
          expAlignCount = np.zeros((len(ts), len(fs), len(ts)))
          for j, fw in enumerate(fs):
            # Start counting at the second word of the foreign sentence
            if fw not in expTransCount.keys():
              expTransCount[str(j)+'_'+fw] = {}
            for k, tw in enumerate(ts):
              if j == 0:
                expInitCount[k] += self.trans[tw][fs[0]]
              else:
                for l, tw in enumerate(ts):
                  expTransCount[str(j)+'_'+fw][str(k)+'_'+tw] += self.alignPriors[l][k] * self.trans[tw][fw]
                  expAlignCount[l][j][k] += self.alignPriors[l][k] * self.trans[tw][fw]

          # Normalization across target words in the sentence
          for i in range(len(ts)):
            expInitCount[i] /= sum(expInitCount) 

          for j, fw in enumerate(fs):
            fKey = str(j)+'_'+fw
            totTransCount = sum(expTransCount[fKey].values())
            totAlignCount = np.sum(expAlignCount, axis=2)
            
            for k, tw in enumerate(ts):
              tKey = str(k)+'_'+tw
              if DEBUG and expTransCount[fKey][tKey] == 0:
                 print(j, k, fw, tw, ts, totTransCount)
              expTransCount[fKey][tKey] /= totTransCount
            
            for l in range(len(ts)):
              expAlignCount[l][j] /= totAlignCount[l][j] 
              
          self.expTransCounts.append(expTransCount)
          self.expAlignCounts.append(expAlignCount)
          self.expInitCounts.append(expInitCount)
    
    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationProbabilities(self):
        # Implement this method
        n_sentence = min(len(self.tCorpus), len(self.fCorpus))
        self.trans = {}
        # Sum all the expected counts across sentence for pairs of translation
        for i in range(n_sentence):
          for j, tw in enumerate(self.tCorpus[i]):
            if tw not in self.trans.keys():
              self.trans[tw] = {}
            for k, fw in  enumerate(self.fCorpus[i]): 
              fKey = str(k)+'_'+fw
              tKey = str(j)+'_'+tw
              for l in range(len(self.tCorpus[i])):
                if fw not in self.trans[tw].keys():
                  self.trans[tw][fw] = self.expTransCounts[i][fKey][tKey] * self.alignPriors[l][j]  
                else:
                  self.trans[tw][fw] += self.expTransCounts[i][fKey][tKey] * self.alignPriors[l][j]  

        # Normalization over all the possible translation of the target word
        for tw in self.trans.keys():
          totCount = sum(self.trans[tw].values())
          for fw in self.trans[tw]:
            #if DEBUG:
            #  print(tw, totCount, self.trans[tw][fw])          
            self.trans[tw][fw] = self.trans[tw][fw] / totCount    
            # Threshold to avoid underflow
            #if self.trans[tw][fw] < PMIN:
            #  self.trans[tw][fw] = PMIN
        #pass
 
    def updateAlignmentProbabilities(self):
      self.alignPriors = {}
      for s, (fs, ts) in enumerate(zip(self.fCorpus, self.tCorpus)):
        for i, tw in enumerate(ts):
          for j, fw in enumerate(fs):
            if j == 0:
              self.alignInit[i] += self.expInitCounts[i] 
            for k in range(len(ts)):
              if k not in self.alignPriors:
                self.alignPriors[k] = {}
                self.alignPriors[k][i] = self.expAlignCounts[s][k][j][i] * self.trans[tw][fw] 
              else:
                self.alignPriors[k][i] += self.expAlignCounts[s][k][j][i] * self.trans[tw][fw] 

        
      for i in self.alignPriors:
        self.alignInit[i] /= sum(self.alignInit.values())
        totCounts = len(self.alignPriors[i].values())
        for j in self.alignPriors[i]:
          self.alignPriors[i][j] /= totCounts  
    
    # Compute average translation probabilities using forward backward algorithm
    def averageTranslationProbability(self):
      avgTransProb = 0.
      if DEBUG:
        print(len(self.fCorpus), self.fCorpus[0])
      for fs, ts in zip(self.fCorpus, self.tCorpus):
        #if DEBUG:
        #  print(ts, fs, len(ts), len(fs))
        avgTransProb += math.log(self.lenProb[len(ts)-1][len(fs)])

        # Assume uniform initial probabilities
        forward = np.zeros((len(ts),))
        for i, tw in enumerate(ts):
          forward[i] = self.alignInit[i] * self.trans[tw][fs[0]]

        # XXX: j is the current time step but fw is the next word
        for j, fw_next in enumerate(fs[1:]):     
          forward_next = np.zeros((len(ts),))  
          for i_next, tw_next in range(len(ts)):
            for i, tw in enumerate(ts):
              forward_next[i_next] += forward[i] * self.alignPriors[i][i_next] * self.trans[tw_next][fw_next]
          forward = deepcopy(forward_next)

        avgTransProb += math.log(sum(forward))
      return avgTransProb / len(self.fCorpus)
    
    def checkConvergence(self, eps=1e-5):
      avgLogTransProb = self.averageTranslationProbability()
      if self.avgLogTransProb == float('-inf'):
        self.avgLogTransProb = avgLogTransProb
        return 0
      if abs((self.avgLogTransProb - avgLogTransProb) / avgLogTransProb) < eps:
        self.avgLogTransProb = avgLogTransProb
        return 1
      
      self.avgLogTransProb = avgLogTransProb  
      return 0
    
    # Returns the best alignment between fSen and tSen using Viterbi algorithm
    # TODO: Find a way to return alignProbs
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###
        alignment = np.zeros((len(tSen), len(fSen)))
        scores = np.zeros((len(tSen),))
        for i, tw in enumerate(tSen):
          scores[i] = self.alignInit[i] * self.trans[tw][fs[0]]

        for j, fw_next in enumerate(fSen[1:]):
          scoresNext = np.zeros((len(tSen),))
          bestScore = 0.  
          for i_next, tw_next in enumerate(tSen):
            transProb = None
            if fw_next not in self.trans[tw_next]:
              obsProb = PMIN  
            else:
              obsProb = self.trans[tw_next][fw_next]

            for i, tw in enumerate(tSen):
              obsProb = None
              # TODO: Need to handle this better (e.g., use backoff)
              if i_next not in self.alignPriors[i]:
                transProb = 1 / len(tSen)
              else:
                transProb = self.alignPriors[i][i_next] 
              
              if bestScore < scores[i] * transProb * obsProb:
                scoresNext[i_next] = scores[i] * transProb * obsProb
                alignment[i_next] = i
          scores = deepcopy(scoresNext) 
            
        bestScore = max(scores)
        endAlign = np.argmax(scores)
        bestAlignment = [endAlign]
        for i in range(len(fSen), 0, -1):
          bestAlignment.append(alignment[bestAlignment[i]]) 
        
        return bestAlignment[::-1]   # Your code above should return the correct alignment instead

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
    def getWordTranslationProbability(self, f_j, e_i):
        # Implement this method
        if e_i in self.trans.keys():
          if f_j in self.trans[e_i].keys():
            return self.trans[e_i][f_j]
          else:
            return float('-inf')
        else:
          return float('-inf')
    
    # Write this model's probability distributions to file
    def printModel(self, filename):
        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        translateProbFile = open(filename+'_translationprobs.txt', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file 
        alignPriorFile = open(filename+'_alignpriors.txt', 'w')
        
        # Implement this method (make your output legible and informative)
        for i in sorted(list(self.alignPriors.keys())):
          for i_next in sorted(list(self.alignPriors[i].keys())):
            alignPriorFile.write('{}\t{}\t{}\t{}\t{}\n'.format(i, i_next, self.alignPriors[i][i_next]))

        for tw in sorted(self.trans.keys()):
          for fw in sorted(self.trans[tw].keys()): 
            translateProbFile.write('{}\t{}\t{}\n'.format(tw, fw, self.trans[tw][fw]))
        
        for tLen in self.lenProb.keys():
          for fLen in self.lenProb[tLen].keys():
            lengthFile.write('{}\t{}\t{}\n'.format(tLen, fLen, self.lenProb[tLen][fLen]))
        #alignProbFile.close()
        lengthFile.close();
        translateProbFile.close()

    # Write the predicted alignment to file
    def printAlignment(self, file_prefix, is_phoneme=True):
      f = open(file_prefix+'.txt', 'w')
      aligns = []
      if DEBUG:
        print(len(self.fCorpus))
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment = self.align(fSen, tSen)
        #if DEBUG:
        #  print(fSen, tSen)
        #  print(alignment)
        align_info = {
            'index': i,
            'image_concepts': tSen, 
            'caption': fSen,
            'alignment': alignment,
            'is_phoneme': is_phoneme
          }
        aligns.append(align_info)
        f.write('%s\n%s\n' % (tSen, fSen))
        for a in alignment:
          f.write('%d ' % a)
        f.write('\n\n')

      f.close()
    
      # Write to a .json file for evaluation
      with open(file_prefix+'.json', 'w') as f:
        json.dump(aligns, f, indent=4, sort_keys=True)            


# utility method to pretty-print an alignment
# You don't have to modify this function unless you don't think it's that pretty...
def prettyAlignment(fSen, tSen, alignment):
    pretty = ''
    for j in range(len(fSen)):
        pretty += str(j)+'  '+fSen[j].ljust(20)+'==>    '+tSen[alignment[j]]+'\n';
    return pretty

if __name__ == "__main__":
    # Initialize model
    #model = IBMModel1('eng-ger.txt')
    datapath = '../data/'
    #model = IBMModel1(datapath + 'mscoco_val.txt')
    #model = IBMModel1(datapath + 'flickr30k/phoneme_level/flickr30k.txt')
    # Train model
    #model.initializeWordTranslationProbabilities(transProbFile='models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt')
    #model.computeTranslationLengthProbabilities()         # <you need to implement computeTranslationlengthProbabilities()>
    
    #print('IBM Model 1 Log likelihood: ', model.averageTranslationProbability())
    #model.trainUsingEM(1, writeModel=True)
    
    #model.computeTranslationLengthProbabilities()
    #model.printModel('after_training')
    #model.printAlignment('mscoco_val_pred_alignment.txt')
    # Use model to get an alignment
    #fSen = 'No pierdas el tiempo por el camino .'.split()
    #tSen = 'Don\' t dawdle on the way'.split()
    #alignment = model.align(fSen, tSen);
    #print(prettyAlignment(fSen, tSen, alignment))
    model = IBMModel2(datapath + 'flickr30k/phoneme_level/flickr30k.txt')
    model.trainUsingEM(writeModel=True)
    #model.trainUsingEM(transProbFile='models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt', writeModel=True)
    #model.initializeWordTranslationProbabilities(transProbFile='models/mar18_flickr30k_phoneme_level_ibm2/model_iter=7.txt_translationprobs.txt')
    #model.initializeAlignmentProbabilities(alignProbFile='models/mar18_flickr30k_phoneme_level_ibm2/model_iter=7.txt_alignpriors.txt')

    model.printAlignment('flickr30k_pred_alignment', is_phoneme=True)
