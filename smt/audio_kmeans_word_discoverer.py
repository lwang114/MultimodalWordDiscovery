import numpy as np
import time
from copy import deepcopy
import json
import random

DEBUG = False
NULL = 'NULL'
random.seed(2)
np.random.seed(2)

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

    def initialize(self, centroidFile=None, initMethod= "kmeans++"):
      if centroidFile:
        with open(centroidFile, 'r') as f:
          self.centroids = json.load(f)
          self.centroids = {tw: np.array(c) for tw, c in self.centroids.items()} 
        return

      # Cyclic intialization
      # TODO: use Kmeans++ later
      if initMethod == "cyclic":
        for tSen, fSen in zip(self.tCorpus, self.fCorpus):
          nframes = fSen.shape[0]
          self.featDim = fSen.shape[1]
          for tw in tSen: 
            for i_f in range(nframes):
              if tw not in self.centroids:
                self.centroids[tw] = np.zeros((self.numMixtures, self.featDim))
              self.centroids[tw][i_f % self.numMixtures] += fSen[i_f] 

      # TODO: Intialize for more than one mixture
      elif initMethod == "kmeans++":
        # Keep a dictionary of candidate centroid vectors according to its co-occurrences with the concept
        candidates = {}
        candidate_counts = {}

        for tSen, fSen in zip(self.tCorpus, self.fCorpus):
          self.featDim = fSen.shape[1]
          for tw in tSen:
            if tw not in candidates:
              candidates[tw] = []
              candidate_counts[tw] = 0
        
        for i_ex, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
          for tw in tSen:
            candidates[tw].append((i_ex, fSen.shape[0]))
            candidate_counts[tw] += fSen.shape[0]
        
        # Randomly samples a subset of candidates for the current centroid
        candidate_subset = {}
        for tw in candidates:
          count = candidate_counts[tw]
          if tw not in candidate_subset:
            candidate_subset[tw] = []
          
          # Randomly draw k frames
          if count <= 100:
            for cand_id, cand in candidates[tw]:
              candidate_subset[tw] += self.fCorpus[cand_id].tolist()
              
            candidate_subset[tw] = np.array(candidate_subset[tw])           
          else:
            rand_ids = np.random.randint(0, count-1, 100)
            for r_id in rand_ids.tolist():
              acc = 0
              for cand_id, cand in candidates[tw]:
                if r_id < acc + cand and r_id >= acc: 
                  candidate_subset[tw].append(self.fCorpus[cand_id][r_id - acc])   
                else:
                  acc += cand
            candidate_subset[tw] = np.array(candidate_subset[tw])
          
        # Compute the distance of frames in the subset to the nearest centroid, 
        # and choose according to a distribution proportional to their distances
        distances = {}
        for i_t, (tw, feats) in enumerate(sorted(candidate_subset.items(), key=lambda x:x[0])):
          if i_t == 0:
            count = feats.shape[0]
            self.centroids[tw] = np.zeros((self.numMixtures, self.featDim))
              
            for m in range(self.numMixtures):
              rand_id = random.randint(0, count-1)
              self.centroids[tw][m] = feats[rand_id]
          else: 
            count = feats.shape[0]
            centroids = self.centroids.values()
            distances[tw] = np.zeros(((i_t+1)*self.numMixtures, count)) 
            self.centroids[tw] = np.zeros((self.numMixtures, self.featDim))
            
            if DEBUG:
              print(centroids) 
            for i_c, cent in enumerate(centroids):
              for m in range(self.numMixtures):
                if DEBUG:
                  print(distances[tw].shape)
                distances[tw][i_c*self.numMixtures+m] = np.sum((feats - cent[m]) ** 2, axis=1)
            
            for m in range(self.numMixtures):
              rand_id = self.randomDraw(np.min(distances[tw], axis=0))
              self.centroids[tw][m] = feats[rand_id]           
        
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
        if DEBUG:
          print(dist_mat)
          print(assignment)
        self.assignments.append(assignment)

    def updateCentroid(self):
      if DEBUG:
        print(self.centroids)  
      self.centroids = {tw:np.zeros(cent.shape) for tw, cent in self.centroids.items()}
      self.counts = {tw:np.zeros((self.numMixtures,)) for tw in self.centroids}
      for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        tLen = len(tSen)
        fLen = fSen.shape[0]
        for i_f, centroid_id in enumerate(self.assignments[i].tolist()):
          i_t = int(centroid_id / self.numMixtures)
          m = centroid_id % self.numMixtures
          if DEBUG:
            print(tSen[i_t])
            #print(fSen.shape)
            #print(self.centroids[tSen[i_t]][m].shape)
            
          self.centroids[tSen[i_t]][m] += fSen[i_f]
          self.counts[tSen[i_t]][m] += 1

      # Normalize the cluster centroids
      for tw in self.centroids:
        for m in range(self.numMixtures):
          if self.counts[tw][m] > 0:
            self.centroids[tw][m] /= self.counts[tw][m]

    def train(self, maxIterations=10, centroidFile=None, modelPrefix='', writeModel=False, initMethod='kmeans++'):
      self.initialize(centroidFile=centroidFile, initMethod=initMethod)
      self.printModel(modelPrefix+'model_init.txt')

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
    
    # TODO: Randomly draw a sample according to a probability mass distribution
    def randomDraw(self, pmf):
      max_val = np.sum(pmf)
      rand_val = max_val * random.random()
      rand_id = 0
      tot = 0.
      while tot < rand_val:
        tot += pmf[rand_id] 
        rand_id += 1
      return rand_id 

    # TODO: Use Bregman divergence other than Euclidean distance (e.g., Itakura divergence)
    def computeDist(self, x, y):
      return np.sqrt(np.sum((x - y)**2))
    
  
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
  datapath = "../data/random/"
  mkmeans = KMeansWordDiscoverer(datapath + "random.npz", datapath + "random.txt", 1)
  mkmeans.train(writeModel=True, modelPrefix="random_", initMethod='cyclic')
  mkmeans.printAlignment("random_pred")
