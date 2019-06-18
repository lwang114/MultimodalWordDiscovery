import numpy as np
import time
from copy import deepcopy
import json

NULL = 'NULL'
class SegEmbedKMeansWordDiscoverer:
    def __init__(self, sourceCorpusFile, targetCorpusFile, numMixtures, embedDim=64, minWordLen=10, maxWordLen=100):
      self.fCorpus = []
      self.tCorpus = []
      self.parseCorpus(sourceCorpusFile, targetCorpusFile)
      
      self.centroids = {}
      self.assignments = []
      self.segmentations = []
      
      self.numMixtures = numMixtures
      self.embedDim = embedDim
      self.minWordLen = minWordLen
      self.maxWordLen = maxWordLen

    # Tokenize the corpus 
    def parseCorpus(self, sourceFile, targetFile):
      fp = open(targetFile, 'r')
      tCorpus = fp.read().split('\n')
      self.tCorpus = [[NULL] + tSen.split() for tSen in tCorpus]
      fCorpus = np.load(sourceFile)
      self.fCorpus = [fCorpus[fKey] for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
      self.data_ids = [fKey for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]

    def initialize(self, centroidFile=None):
      if centroidFile:
        with open(centroidFile, 'r') as f:
          self.centroids = json.load(f)
          self.centroids = {tw: np.array(c) for tw, c in self.centroids.items()} 
        return

      # Cyclic intialization
      # TODO: use Kmeans++ later
      for tSen, fSen in zip(self.tCorpus, self.fCorpus):
        nFrames = fSen.shape[0]
        # TODO: Try skipping
        nSegments = nFrames
        self.featDim = fSen.shape[1]
        for tw in tSen: 
          for i_f in range(nSegments):
            if tw not in self.centroids:
              self.centroids[tw] = np.zeros((self.numMixtures, self.embedDim))
            if i_f + int(self.embedDim / self.featDim) > nFrames:
              continue
            self.centroids[tw][i_f % self.numMixtures] += fSen[i_f:i_f+int(self.embedDim / self.featDim)].flatten() 
        
    # TODO: Use Bregman divergence other than Euclidean distance (e.g., Itakura divergence)
    def computeDist(self, x, y):
      return np.sqrt(np.sum((x - y)**2))
    
    def findAssignment(self):
      self.assignments = []
      for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        segmentation = self.segmentations[i] 
        assignment = self.assign(fSen, tSen, segmentation)
        self.assignments.append(assignment)

    def updateCentroid(self):  
      self.centroids = {tw:np.zeros(cent.shape) for tw, cent in self.centroids.items()}
      self.counts = {tw:np.zeros((self.numMixtures,)) for tw in self.centroids}
      for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        tLen = len(tSen)
        fLen = fSen.shape[0]
        segmentation = self.segmentations[i]
        for i_w, centroid_id in enumerate(self.assignments[i]):
          print(self.assignments[i], segmentation)
          i_t = int(centroid_id / self.numMixtures)
          m = centroid_id % self.numMixtures
          start, end = None, None
          if i_w == 0:
            start, end = 0, segmentation[i_w]+1
          else:
            start, end = segmentation[i_w-1]+1, segmentation[i_w]+1
          fEmbed = self.embed(fSen[start:end])
          self.centroids[tSen[i_t]][m] += fEmbed
          self.counts[tSen[i_t]][m] += 1

      # Normalize the cluster centroids
      for tw in self.centroids:
        for m in range(self.numMixtures):
          if self.counts[tw][m] > 0:
            # Approximate update formula; exact should be weighted by the duration 
            self.centroids[tw][m] /= self.counts[tw][m]

    def kMeansStep(self, maxIterations=10):
      n_iter = 0
      prev_assignment = deepcopy(self.assignments)
      while (n_iter < maxIterations and not self.checkConvergence(prev_assignment, self.assignments)) or n_iter == 0:
        prev_assignment = deepcopy(self.assignments)
        print("Starting training iteration "+str(n_iter))
        begin_time = time.time()
        self.findAssignment()
        print('Assignment step takes %0.5f s to finish' % (time.time() - begin_time))
        
        begin_time = time.time()
        self.updateCentroid()
        print('Update step takes %0.5f s to finish' % (time.time() - begin_time))
        
        n_iter += 1

    def segmentStep(self):
      assert self.minWordLen * self.featDim >= self.embedDim 
      self.segmentations = []
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        segmentation = self.segment(fSen, tSen, self.minWordLen, self.maxWordLen)
        self.segmentations.append(segmentation)

    def trainUsingEM(self, maxIterations=10, centroidFile=None, modelPrefix='', writeModel=False):
      self.initialize(centroidFile=centroidFile)

      prev_assignments = deepcopy(self.assignments)
      n_iter = 0
      
      while (n_iter < maxIterations and not self.checkConvergence(prev_assignments, self.assignments)) or n_iter == 0:
        prev_assignments = deepcopy(self.assignments)
      
        begin_time = time.time()
        self.segmentStep()
        print('Segment step takes %0.5f to finish' % (time.time() - begin_time))

        begin_time = time.time()
        self.kMeansStep()
        print('K-Means step takes %0.5f to finish' % (time.time() - begin_time))
        
        if writeModel:
          self.printModel(modelPrefix+'model_iter='+str(n_iter)+'.txt')
        
        n_iter += 1

    def checkConvergence(self, prevAssigns, curAssigns):
      for prev_assign, cur_assign in zip(prevAssigns, curAssigns):
        if not (prev_assign == cur_assign).all():
          return 0 
      return 1

    def printModel(self, filename):
      with open(filename, 'w') as f:
        centroids = {tw: c.tolist() for tw, c in self.centroids.items()}
        json.dump(centroids, f)
    
    # Embed a segment into a fixed-length vector 
    def embed(self, x):
      assert self.embedDim % self.featDim == 0
      xLen = x.shape[0]
      skip = int(xLen / (self.embedDim / self.featDim))
      #print(xLen, self.embedDim / self.featDim)
      return x[::skip].flatten()[:self.embedDim]

    def assign(self, fSen, tSen, segmentation):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      numWords = len(segmentation)
      dist_mat = np.zeros((tLen, self.numMixtures, numWords))
      for i_w in range(numWords):
        for i_t, tw in enumerate(tSen):
          for m in range(self.numMixtures):
            seg = None
            if i_w == 0:
              seg = fSen[:segmentation[0]+1]
            else:
              seg = fSen[segmentation[i_w-1]+1:segmentation[i_w]+1]
            
            dist_mat[i_t, m, i_w] = self.computeDist(self.embed(seg), self.centroids[tw][m])
      assignment = np.argmin(dist_mat.reshape(-1, numWords), axis=0)
      print(assignment)
      return assignment        

    def segment(self, fSen, tSen, minWordLen, maxWordLen):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      segmentCosts = [np.finfo(float).max]*fLen
      segmentPaths = [0]*fLen
      for i_f in range(minWordLen-1, fLen):
        for j_f in range(minWordLen, maxWordLen+1):
          dists = np.zeros((tLen, self.numMixtures))  
          if i_f - j_f + 1 == 0:
            for i_t, tw in enumerate(tSen): 
              for m in range(self.numMixtures):
                # Distance weighted by the number of frames represented by the embedding
                #dists[i_t, m] = j_f * self.computeDist(self.embed(fSen[i_f-j_f+1:i_f+1]), self.centroids[tw][m])  
                # Unweighted distance
                dists[i_t, m] = self.computeDist(self.embed(fSen[i_f-j_f+1:i_f+1]), self.centroids[tw][m])  

            segmentCosts[i_f] = np.min(dists)
          elif i_f - j_f + 1 < minWordLen:
            continue
          else:
            for i_t, tw in enumerate(tSen): 
              for m in range(self.numMixtures):
                # Distance weighted by the number of frames
                #dists[i_t, m] = j_f * self.computeDist(self.embed(fSen[i_f-j_f+1:i_f+1]), self.centroids[tw][m])  
                # Unweighted distance
                dists[i_t, m] = self.computeDist(self.embed(fSen[i_f-j_f+1:i_f+1]), self.centroids[tw][m])  

            curCost = segmentCosts[i_f - j_f] + np.min(dists) 
            if curCost < segmentCosts[i_f]:
              segmentCosts[i_f] = curCost
              segmentPaths[i_f] = i_f - j_f
      
      # Follow the back pointers to find the optimal segmentation
      i_f = fLen - 1
      #print('segmentCosts: ', segmentCosts)
      #print('segmentPaths: ', segmentPaths)
      best_segmentation = [i_f]
      
      while segmentPaths[i_f] != 0:
        i_f = segmentPaths[i_f]
        best_segmentation.append(i_f)
      return best_segmentation[::-1]

    def align(self, fSen, tSen):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      alignment = [0]*fLen
      segmentation = self.segment(fSen, tSen, self.minWordLen, self.maxWordLen)
      assignment = self.assign(fSen, tSen, segmentation)
      numWords = len(segmentation)
      for i_w in range(numWords):
        start, end = None, None
        if i_w == 0:
          start, end = 0, segmentation[i_w]+1
        else:
          start, end = segmentation[i_w-1]+1, segmentation[i_w]+1

        alignment[start:end] = [int(assignment[i_w] / self.numMixtures)] * (end - start) 
      return alignment

    def printAlignment(self, filePrefix):
      f = open(filePrefix+'.txt', 'w')
      aligns = []
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment = self.align(fSen, tSen)
        align_info = {
          'index': self.data_ids[i],
          'image_concepts': tSen,
          'alignment': alignment,
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
  #datapath = '../data/flickr30k/audio_level/'
  #src_file = 'flickr_bnf_all_src.npz'
  #trg_file = 'flickr_bnf_all_trg.txt'
  datapath = "../"
  src_file = "random.npz"
  trg_file = "random.txt"

  model = SegEmbedKMeansWordDiscoverer(datapath+src_file, datapath+trg_file, 1, 20)
  model.trainUsingEM(writeModel=True, modelPrefix="random_kmeans_embed_")
  model.printAlignment("random_kmeans_embed_pred")
