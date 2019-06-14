import numpy as np
import time
from copy import deepcopy
import json

NULL = 'NULL'
class KMeansWordDiscoverer:
    def __init__(self, sourceCorpusFile, targetCorpusFile, numMixtures):
      self.fCorpus = []
      self.tCorpus = []
      self.parseCorpus(sourceCorpusFile, targetCorpusFile)
      
      self.centroids = {}
      self.assignments = []
      
      self.numMixtures = numMixtures

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
        nframes = fSen.shape[0]
        self.featDim = fSen.shape[1]
        for tw in tSen: 
          for i_f in range(nframes):
            if tw not in self.centroids:
              self.centroids[tw] = np.zeros((self.numMixtures, self.featDim))
            
            self.centroids[tw][i_f % self.numMixtures] += fSen[i_f] 
    
    # TODO: Use Bregman divergence other than Euclidean distance (e.g., Itakura divergence)
    def computeDist(self, x, y):
      return np.sqrt(np.sum((x - y)**2))
    
    def findAssignment(self):
      self.assignments = []
      for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        fLen = fSen.shape[0]
        tLen = len(tSen)
        dist_mat = np.zeros((tLen, self.numMixtures, fLen))
        for i_f in range(fLen):
          for i_t, tw in enumerate(tSen):
            for m in range(self.numMixtures):
              dist_mat[i_t, m, i_f] = self.computeDist(fSen[i_f], self.centroids[tw][m])
        assignment = np.argmin(dist_mat.reshape(-1, fLen), axis=0)
        
        self.assignments.append(assignment)

    def updateCentroid(self):  
      self.centroids = {tw:np.zeros(cent.shape) for tw, cent in self.centroids.items()}
      self.counts = {tw:np.zeros((self.numMixtures,)) for tw in self.centroids}
      for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        tLen = len(tSen)
        fLen = fSen.shape[0]
        for i_f, centroid_id in enumerate(self.assignments[i].tolist()):
          i_t = int(centroid_id / self.numMixtures)
          m = centroid_id % self.numMixtures
          self.centroids[tSen[i_t]][m] += fSen[i_f]
          self.counts[tSen[i_t]][m] += 1

      # Normalize the cluster centroids
      for tw in self.centroids:
        for m in range(self.numMixtures):
          if self.counts[tw][m] > 0:
            self.centroids[tw][m] /= self.counts[tw][m]

    def trainUsingEM(self, maxIterations=10, centroidFile=None, modelPrefix='', writeModel=False):
      self.initialize(centroidFile=centroidFile)

      prev_assignments = deepcopy(self.assignments)
      n_iter = 0
      
      while (n_iter < maxIterations and not self.checkConvergence(prev_assignments, self.assignments)) or n_iter == 0:
        print("Starting training iteration "+str(n_iter))
        begin_time = time.time()
        self.findAssignment()
        print('Assignment step takes %0.5f s to finish' % (time.time() - begin_time))
        
        begin_time = time.time()
        self.updateCentroid()
        print('Update step takes %0.5f s to finish' % (time.time() - begin_time))

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
      
    def align(self, fSen, tSen):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      alignment = [0]*fLen
      for i_f in range(fLen):
        dists = np.zeros((tLen, self.numMixtures))
        for i_t, tw in enumerate(tSen):
          for m in range(self.numMixtures):
            dists[i_t, m] = self.computeDist(fSen[i_f], self.centroids[tw][m])
        alignment[i_f] = int(np.argmin(np.min(dists, axis=1)))
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

if __name__ == "__main__":
  mkmeans = KMeansWordDiscoverer("../random.npz", "../random.txt", 1)
  mkmeans.trainUsingEM(writeModel=True, modelPrefix="random_")
  mkmeans.printAlignment("random_pred")
