import math
import json
import numpy as np
import time
from smt_word_discoverer import IBMModel1, IBMModel2
from nltk.tokenize import word_tokenize

# Constant for NULL word at position zero in target sentence
NULL = "NULL"
UNKPROB = 1e-12
DEBUG = False

class StatisticalPhonemeImageRetriever:
    def __init__(self, trainingCorpusFile, testCorpusFile, 
                       modelType='ibm1', 
                       transProbFile=None, alignProbFile=None, modelPath=None):
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences
        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences
        self.modelType = modelType
        self.modelPath = modelPath

        if modelType == 'ibm1':
          self.aligner = IBMModel1(trainingCorpusFile)
          self.aligner.initializeWordTranslationProbabilities(transProbFile)
          # Smooth the length probabilities
          self.aligner.computeTranslationLengthProbabilities(smoothing='laplace')
        elif modelType == 'ibm2':
          self.aligner = IBMModel2(trainingCorpusFile)
          self.aligner.initializeWordTranslationProbabilities(transProbFile)
          self.aligner.initializeAlignmentProbabilities(alignProbFile)

        self.initialize(testCorpusFile)
        self.queries = self.fCorpus
        self.database = self.tCorpus
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
        self.aligner.trainUsingEM(numIterations, writeModel, epsilon=epsilon)
        self.aligner.printModel(self.modelPath)

    # Returns the best alignment between fSen and tSen, according to your model
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###
        alignment = [0]*len(fSen)
        bestSentProb = 1. # TODO: Use log-prob later
        for i, fw in enumerate(fSen):
          bestProb = float('-inf')
          for j, tw in enumerate(tSen):
            if tw not in self.aligner.trans:
              bestProb = UNKPROB
              break
            elif fw not in self.aligner.trans[tw]:
              continue

            score = 0.
            if self.modelType == 'ibm1':
              if len(tSen) not in self.aligner.lenProb:
                score = UNKPROB
                break
              elif len(fSen) not in self.aligner.lenProb[len(tSen)]:
                continue
              score = self.aligner.lenProb[len(tSen)][len(fSen)] * self.aligner.trans[tw][fw]
            elif self.modelType == 'ibm2':
              if len(tSen) not in self.aligner.alignPriors:
                score = UNKPROB
                break
              elif len(fSen) not in self.aligner.alignPriors[len(tSen)]:
                continue
              score = self.aligner.alignPriors[len(tSen)][len(fSen)][j][i] * self.aligner.trans[tw][fw]  
              
            if score > bestProb:
              alignment[i] = j
              bestProb = score 
            
          bestSentProb *= bestProb
        # Length Normalization
        #bestSentProb *= self.getLengthProb(len(fSen), len(tSen)) #self.aligner.getTranslationLengthProbability(len(fSen), len(tSen)) 
        return alignment, bestSentProb       
    
    #def getLengthProb(self, fLen, tLen, lbda=1.09):
    #    return (lbda * tLen) ** (fLen) * math.exp(- lbda * tLen) / math.factorial(fLen) 

    def retrieve(self, query, nbest=10):
        scores = []
        for tSen in self.tCorpus:
          score = self.align(query, tSen)[1]
          scores.append(score)
        if DEBUG:
          print(scores)
        kbest = np.argsort(np.array(scores))[-nbest:][::-1]
        return kbest.tolist()

    def evaluate(self, out_file):
      recall_at_1 = 0.
      recall_at_5 = 0.
      recall_at_10 = 0.
      n = len(self.queries)
      kbests = []
      for i, query in enumerate(self.queries):
        kbest = self.retrieve(query, 10)
        kbest_str = [str(k) for k in kbest]
        kbests.append(' '.join(kbest_str))
        if DEBUG:
          print(kbest)
        if kbest[0] == i:
          recall_at_1 += 1
          recall_at_5 += 1
          recall_at_10 += 1
          continue
        found = 0
        for j in kbest[:5]:
          if i == j:
            recall_at_5 += 1
            recall_at_10 += 1
            found = 1
            break
        if found:
          continue
        
        for j in kbest:
          if i == j:
            recall_at_10 += 1
      
      recall_at_1 /= n
      recall_at_5 /= n
      recall_at_10 /= n
      print('Recall@1: ', recall_at_1)
      print('Recall@5: ', recall_at_5)
      print('Recall@10: ', recall_at_10) 
      
      with open(out_file, 'w') as f:
        f.write('\n'.join(kbests))

      kbest_concepts = []
      with open(out_file+'.readable', 'w') as f:
        for kbest_str in kbests:
          kbest_res = [' '.join(self.database[int(k)]) for k in kbest_str.split()]
          kbest_res = '\n'.join(kbest_res)
          kbest_concepts.append(kbest_res)
        f.write('\n\n'.join(kbest_concepts))

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
    start = time.time()
    datapath = '../data/'
    start = time.time()
    '''model = StatisticalPhonemeImageRetriever(datapath + 'flickr30k/phoneme_level/flickr30k.train',
                                  datapath + 'flickr30k/phoneme_level/flickr30k.test', 
                                  modelType = 'ibm2',
                                  transProbFile='models/mar18_flickr30k_phoneme_level_ibm2/model_iter=7.txt_translationprobs.txt',
                                  alignProbFile='models/mar18_flickr30k_phoneme_level_ibm2/model_iter=7.txt_alignpriors.txt',
                                  modelPath='models/mar18_flickr30k_retrieval_ibm2/') 
    model.trainUsingEM(50, writeModel=True)
    print('Time spend on training: ', time.time() - start)
    start = time.time()
    model.evaluate('exp/mar18_flickr30k_retrieval_ibm2/output/pred_kbest.txt')
    print('Time spent on retrieval: ', time.time() - start)
    '''
    #model = IBMModel1(datapath + 'mscoco_val.txt')
    model = StatisticalPhonemeImageRetriever(datapath + 'flickr30k/phoneme_level/flickr30k.train',
                                  datapath + 'flickr30k/phoneme_level/flickr30k.test', 
                                  modelType = 'ibm1',
                                  transProbFile='models/mar14th_retrieval/model_iter=48.txt_translationprobs.txt')
                                  #alignProbFile='models/mar18_flickr30k_phoneme_level_ibm2/model_iter=7.txt_alignpriors.txt',
                                  #modelPath='models/mar18_flickr30k_retrieval_ibm1_smoothing') 
    #pretrainModel='models/flickr30k_phoneme_level/model_iter=46.txt_translationprobs.txt')
    print('Time spend on training: ', time.time() - start)
    start = time.time()
    model.evaluate('exp/mar18_flickr30k_retrieval_ibm1_smoothing/output/pred_kbest.txt')
    print('Time spent on retrieval: ', time.time() - start)
