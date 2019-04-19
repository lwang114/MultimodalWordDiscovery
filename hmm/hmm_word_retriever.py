from hmm_word_discoverer import *
import numpy as np

DEBUG = True
class HMMWordRetriever:
  def __init__(self, trainingCorpusFile, initProbFile=None, transProbFile=None, obsProbFile=None, modelName='hmm_word_retriever'):
    self.modelName = modelName
    self.aligner = HMMWordDiscoverer(trainingCorpusFile, initProbFile, transProbFile, obsProbFile, modelName=modelName)    
    self.imgc_database = self.aligner.tCorpus
    self.phn_database = self.aligner.fCorpus 

  def train(self, numIterations=10, writeModel=False):
    self.aligner.trainUsingEM(numIterations, writeModel=True) 

  ### Perform two tasks:
  #   * Image search: given the phonemes, generate the image concepts
  #   * Captioning: given a image concepts, generate the phonemes
  def retrieve(self, query, kbest=10, imageSearch=True):
    scores = []
    paths = []
    if imageSearch:
      for imgc in self.imgc_database:
        score, best_path = self.aligner.align(query, imgc)
        scores.append(score)
        paths.append(best_path)
    else:
      for phn in self.phn_database:
        score, best_path = self.aligner.align(phn, query)
        scores.append(score)
        paths.append(best_path)
    
    kbest_indices = sorted(scores)[-kbest:][::-1]
    best_paths = [paths[i] for i in kbest_indices]
    return kbest_indices, best_paths

  def retrieve_all(self):       
    assert len(self.aligner.tCorpus) == len(self.aligner.fCorpus)
    n = len(self.aligner.tCorpus)
    scores = np.zeros((n, n))
    for phn_id, phn in enumerate(self.phn_database):
      for img_id, imgc in enumerate(self.imgc_database):
        if DEBUG:
          print('phn, imgc: ', phn, imgc)
        best_path, score = self.aligner.align(phn, imgc)
        scores[phn_id][img_id] = score
    return scores

  def evaluate(self, kbest=10, outFile=None):
    scores = self.retrieve_all()
    I_kbest = np.argsort(scores, axis=1)[-kbest:][::-1]
    P_kbest = np.argsort(scores, axis=0)[-kbest:][::-1]
    n = len(scores)
    I_recall_at_1 = 0.
    I_recall_at_5 = 0.
    I_recall_at_10 = 0.
    P_recall_at_1 = 0.
    P_recall_at_5 = 0.
    P_recall_at_10 = 0.

    for i in range(n):
      if I_kbest[i][0] == i:
        I_recall_at_1 += 1
      
      for j in I_kbest[i][:5]:
        if i == j:
          I_recall_at_5 += 1
       
      for j in I_kbest[i][:10]:
        if i == j:
          I_recall_at_10 += 1
      
      if P_kbest[0][i] == i:
        P_recall_at_1 += 1
      
      for j in P_kbest[:5, i]:
        if i == j:
          P_recall_at_5 += 1
       
      for j in P_kbest[:10, i]:
        if i == j:
          P_recall_at_10 += 1

    I_recall_at_1 /= n
    I_recall_at_5 /= n
    I_recall_at_10 /= n
    P_recall_at_1 /= n
    P_recall_at_5 /= n
    P_recall_at_10 /= n
     
    print('Image Search Recall@1: ', I_recall_at_1)
    print('Image Search Recall@5: ', I_recall_at_5)
    print('Image Search Recall@10: ', I_recall_at_10)
    print('Captioning Recall@1: ', P_recall_at_1)
    print('Captioning Recall@5: ', P_recall_at_5)
    print('Captioning Recall@10: ', P_recall_at_10)

    fp1 = open('image_search_'+outFile, 'w')
    fp2 = open('image_search_'+outFile+'.readable', 'w')
    for i in range(n):
      I_kbest_str = ' '.join([str(idx) for idx in I_kbest[i]])
      imgcs = '\n'.join([' '.join(self.imgc_database[i]) for i in I_kbest[i]])
        
      fp1.write(I_kbest_str + '\n')
      fp2.write(imgcs + '\n')
    fp1.close()
    fp2.close() 

    fp1 = open('captioning_'+outFile, 'w')
    fp2 = open('captioning_'+outFile+'.readable', 'w')
    for i in range(n):
      P_kbest_str = ' '.join([str(idx) for idx in P_kbest[i]])
      phns = '\n'.join([' '.join(self.phn_database[i]) for i in P_kbest[i]])
        
      fp1.write(P_kbest_str + '\n')
      fp2.write(phns + '\n')
    fp1.close()
    fp2.close()
       
if __name__ == '__main__':
  trainingCorpusFile = 'test_translation.txt' 
  #'../data/flickr30k/phoneme_level/flickr30k.txt'
  initProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_initialprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_initialprobs.txt'
  transProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_transitionprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_transitionprobs.txt'
  obsProbFile = 'models/apr18_tiny_hmm_translate/A_iter=9.txt_observationprobs.txt'
  #'hmm_word_discoverer_iter=9.txt_observationprobs.txt'

  model = HMMWordRetriever(trainingCorpusFile, modelName='A')
  #model = HMMWordRetriever(trainingCorpusFile, initProbFile, transProbFile, obsProbFile, modelName='A')
  model.train()
  #model.aligner.initializeModel()
  model.evaluate(outFile='A.out') 
