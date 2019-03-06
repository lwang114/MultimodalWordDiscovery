import math
import json
from nltk.tokenize import word_tokenize
# Constant for NULL word at position zero in target sentence
NULL = "NULL"
DEBUG = True
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
    def trainUsingEM(self, numIterations=10, writeModel=False, epsilon=1e-5):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities()         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationProbabilities()        # <you need to implement initializeTranslationProbabilities()>
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
    def initializeWordTranslationProbabilities(self, trans_prob_file=None):
        # Implement this method
        self.trans = {}
        if trans_prob_file:
          f = open(trans_prob_file)
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
              # Exponentiate the log translation probability to compute expectation
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
    
    def averageTranslationProbability(self):
      avgTransProb = 0.
      if DEBUG:
        print(len(self.fCorpus), self.fCorpus[0])
      for fs, ts in zip(self.fCorpus, self.tCorpus):
        #if DEBUG:
        #  print(ts, fs, len(ts), len(fs))
           
        avgTransProb += math.log(self.lenProb[len(ts)-1][len(fs)]) + len(fs) * math.log(len(ts))
        for fw in fs:   
          avgTransWord = 0.
          for tw in ts:
            avgTransWord += self.trans[tw][fw]
          avgTransProb += math.log(avgTransWord)
      return avgTransProb
      
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
          
    # Returns the best alignment between fSen and tSen, according to your model
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###
        alignment = [0]*len(fSen)
        for i, fw in enumerate(fSen):
          bestProb = float('-inf')
          for j, tw in enumerate(tSen):
            if self.trans[tw][fw] > bestProb:
              alignment[i] = j
              bestProb = self.trans[tw][fw]
        return alignment   # Your code above should return the correct alignment instead

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
        #alignProbFile = open(filename+'_alignprobs.txt', 'w')
        # Implement this method (make your output legible and informative)
        #for i, f_dict in enumerate(self.alignProb):
        #  for fw in f_dict.keys():
        #    for tw in f_dict[fw].keys():
        #      alignProbFile.write('{}\t{}\t{}\t{}\n'.format(i, fw, tw, self.alignProb[i][fw][tw]))

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
    model = IBMModel1(datapath + 'flickr30k/phoneme_level/flickr30k.txt')
    # Train model
    #model.trainUsingEM(20, writeModel=True)
    model.initializeWordTranslationProbabilities(trans_prob_file='models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt')
    #model.computeTranslationLengthProbabilities()
    #model.printModel('after_training')
    #model.printAlignment('mscoco_val_pred_alignment.txt')
    model.printAlignment('flickr30k_pred_alignment.txt', is_phoneme=True)

    # Use model to get an alignment
    #fSen = 'No pierdas el tiempo por el camino .'.split()
    #tSen = 'Don\' t dawdle on the way'.split()
    #alignment = model.align(fSen, tSen);
    #print(prettyAlignment(fSen, tSen, alignment))
