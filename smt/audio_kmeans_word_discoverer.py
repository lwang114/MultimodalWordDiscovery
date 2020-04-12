import numpy as np
import time
from copy import deepcopy
import json
import random

DEBUG = False
NULL = 'NULL'
ORD = 'C'
#random.seed(2)
#np.random.seed(2)

# Randomly draw a sample according to a probability mass distribution
def randomDraw(pmf):
  max_val = np.sum(pmf)
  rand_val = max_val
  while rand_val >= max_val:
    rand_val = max_val * random.random()
  
  rand_id = 0
  tot = pmf[0]
  while tot < rand_val:
    rand_id += 1
    tot += pmf[rand_id] 
    
  return rand_id 

class KMeansWordDiscoverer:
    def __init__(self, numMixtures, sourceCorpusFile=None, targetCorpusFile=None, 
                contextWidth=0, fCorpus=None, tCorpus=None, 
                centroidFile=None, initMethod="rand"):
      self.fCorpus = fCorpus
      self.tCorpus = tCorpus
       
      if sourceCorpusFile and targetCorpusFile:
        self.parseCorpus(sourceCorpusFile, targetCorpusFile, contextWidth)
      else:
        self.data_ids = list(range(len(fCorpus)))
      
      self.featDim = self.fCorpus[0].shape[1]    
      self.centroids = {}
      self.assignments = []
      self.numMembers = {}
      self.numMixtures = numMixtures

      self.initialize(centroidFile=centroidFile, initMethod=initMethod)

    # Tokenize the corpus 
    def parseCorpus(self, sourceFile, targetFile, contextWidth, maxLen=1000):
      fp = open(targetFile, 'r')
      tCorpus = fp.read().split('\n')
      self.tCorpus = [[NULL] + tSen.split() for tSen in tCorpus]
      fCorpus = np.load(sourceFile)
      self.fCorpus = [concatContext(fCorpus[fKey], contextWidth=contextWidth) for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]
      self.fCorpus = [fSen[:maxLen] for fSen in self.fCorpus]
      self.data_ids = [fKey for fKey in sorted(fCorpus.keys(), key=lambda x:int(x.split('_')[-1]))]

    def initialize(self, centroidFile=None, initMethod="rand"):
      if centroidFile:
        with open(centroidFile, 'r') as f:
          self.centroids = json.load(f)
          self.centroids = {tw: np.array(c) for tw, c in self.centroids.items()} 
        return         

      # Initialize assignment      
      sent_ids = []
      concept2frame = {}
      tot_frames = 0

      for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        # Make sure we have consecutive values
        tLen = len(tSen)
        fLen = fSen.shape[0]
        sent_ids += [(i, k_f) for k_f in range(fLen)]
                
        #init_assignment = np.zeros((fLen,), dtype=int)
        init_assignment = np.random.randint(0, tLen * self.numMixtures, size=(fLen,))
                        
        for k_t, tw in enumerate(tSen):
          for m in range(self.numMixtures):
            if tw not in self.numMembers: 
              self.numMembers[tw] = np.zeros((self.numMixtures,))
              concept2frame[tw] = []
              
            indices_for_tw = np.arange(fLen)
            concept2frame[tw] += (tot_frames + indices_for_tw).tolist()
            self.numMembers[tw][m] += indices_for_tw.shape[0]
        
        #self.assignments.append(np.zeros((fSen.shape[0],)))
        self.assignments.append(init_assignment)
        tot_frames += fLen

      if DEBUG: 
        print("len(self.fCorpus) in kmeans: ", len(self.fCorpus))
        print("len(self.assignments) in kmeans: ", len(self.assignments))

      if initMethod == "kmeans++": 
        # Keep a dictionary of candidate centroid vectors according to its co-occurrences with the concept
        candidates = {}
        candidate_counts = {}

        for tSen, fSen in zip(self.tCorpus, self.fCorpus):
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
          if tw not in candidate_subset:
            candidate_subset[tw] = []
          
          # Randomly draw k frames
          if candidate_counts[tw] <= 100:
            for cand_id, cand in candidates[tw]:
              candidate_subset[tw] += self.fCorpus[cand_id].tolist()
              
            candidate_subset[tw] = np.array(candidate_subset[tw])           
          else:
            rand_ids = np.random.randint(0, candidate_counts[tw]-1, 100)
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
        for i_t, (tw, rand_frames) in enumerate(sorted(candidate_subset.items(), key=lambda x:x[0])):
          # print("initialize centroid %d" % i_t)
          if i_t == 0:
            n_rand_frames = rand_frames.shape[0]
            self.centroids[tw] = np.zeros((self.numMixtures, self.featDim))
              
            for m in range(self.numMixtures):
              self.centroids[tw][m] = rand_frames[random.randint(0, n_rand_frames-1)]
          else: 
            n_rand_frames = rand_frames.shape[0]
            centroids = self.centroids.values()
            distances = np.zeros((i_t*self.numMixtures, n_rand_frames)) 
            self.centroids[tw] = np.zeros((self.numMixtures, self.featDim))
            
            for i_c in range(i_t):
              for m in range(self.numMixtures):
                centroid = centroids[i_c][m]
                if DEBUG:
                  print("i_c, m: %d %d" % (i_c, m))
                  print("distance id: %d" % (i_c*self.numMixtures+m))
                  
                distances[i_c*self.numMixtures+m] = np.sum((rand_frames - centroid) ** 2, axis=1)
            
            if DEBUG:
              print("distances.shape", distances.shape)
              print("rand_frames.shape", rand_frames.shape)
            for m in range(self.numMixtures):
              rand_idx = randomDraw(np.amin(distances, axis=0))
              self.centroids[tw][m] = rand_frames[rand_idx]           
      elif initMethod == "rand":
        for k, (tw, frame_ids) in enumerate(concept2frame.items()):
          frame_ids = np.asarray(frame_ids)
          rand_frame_ids = frame_ids[np.random.randint(0, len(frame_ids)-1, size=(self.numMixtures,)).tolist()]
          if tw not in self.centroids:
            self.centroids[tw] = np.zeros((self.numMixtures, self.featDim))

          for m, frame_id in enumerate(rand_frame_ids.tolist()):
            if DEBUG:
              print("frame_id:", frame_id)
              print("len(sent_ids):", len(sent_ids))
            self.centroids[tw][m] = self.fCorpus[sent_ids[frame_id][0]][sent_ids[frame_id][1]]
      
    def findAssignment(self):
      prev_assignments = deepcopy(self.assignments)
      self.assignments = []
      avgCentroidDistance = 0.
      self.numAssignChanges = 0
      for i, (tSen, fSen) in enumerate(zip(self.tCorpus, self.fCorpus)):
        fLen = fSen.shape[0]
        tLen = len(tSen)
        dist_mat = np.zeros((tLen, self.numMixtures, fLen))
        for i_f in range(fLen):
          for i_t, tw in enumerate(tSen):
            for m in range(self.numMixtures):
              dist_mat[i_t, m, i_f] = self.computeDist(fSen[i_f], self.centroids[tw][m])
        assignment = np.argmin(dist_mat.reshape(-1, fLen), axis=0)
        self.numAssignChanges += np.sum(prev_assignments[i] != assignment) 
        avgCentroidDistance += 1. / len(self.fCorpus) * np.mean(np.amin(dist_mat, axis=(0, 1))) 

        if DEBUG:
          print(dist_mat)
          print(assignment)
        self.assignments.append(assignment)
        self.avgCentroidDistance = avgCentroidDistance

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

    def trainUsingEM(self, numIterations=100, modelPrefix='', writeModel=False):
      prev_assignments = deepcopy(self.assignments)
      n_iter = 0
      
      while (n_iter < numIterations and not self.checkConvergence(prev_assignments, self.assignments)) or n_iter == 0:
        print("Starting training iteration "+str(n_iter))
        begin_time = time.time()
        self.findAssignment()
        print('Assignment step takes %0.5f s to finish' % (time.time() - begin_time))
        print("Number of changes in assignment: ", self.numAssignChanges)
       
        begin_time = time.time()
        self.updateCentroid()
        print('Update step takes %0.5f s to finish' % (time.time() - begin_time))
        print("Average centroid distance: ", self.avgCentroidDistance)

        if writeModel:
          self.printModel(modelPrefix+'model_iter='+str(n_iter)+'.json')
        
        n_iter += 1
      
      if writeModel:
        self.printModel(modelPrefix+'model_final.json')

    def checkConvergence(self, prevAssigns, curAssigns):
      for prev_assign, cur_assign in zip(prevAssigns, curAssigns):
        if not (prev_assign == cur_assign).all():
          return 0 
      return 1

    def printModel(self, filename):
      with open(filename, 'w') as f:
        centroids = {tw: c.tolist() for tw, c in self.centroids.items()}
        json.dump(centroids, f)
   
    # TODO: Use Bregman divergence other than Euclidean distance (e.g., Itakura divergence)
    def computeDist(self, x, y):
      return np.sqrt(np.sum((x - y)**2))  

    def reassign(self, sentId, newSent, newAssigns):
      oldSent = self.fCorpus[sentId]
      tSen = self.tCorpus[sentId]
      oldAssigns = self.assignments[sentId]
      for old_frame, old_k_m in zip(oldSent.tolist(), oldAssigns.tolist()):
        k_t_old = int(old_k_m / self.numMixtures)
        m_old = int(old_k_m % self.numMixtures)
        tw_old = tSen[k_t_old]
         
        newPrevCentroid = self.centroids[tw_old][m_old] * self.numMembers[tw_old][m_old] - np.asarray(old_frame)
        if self.numMembers[tw_old][m_old] <= 0:
          continue
        elif self.numMembers[tw_old][m_old] <= 1:
          self.numMembers[tw_old][m_old] -= 1
          continue
        else:
          self.numMembers[tw_old][m_old] -= 1
          self.centroids[tw_old][m_old] = newPrevCentroid / self.numMembers[tw_old][m_old]

      for new_frame, new_k_m in zip(newSent.tolist(), newAssigns.tolist()): 
        k_t_new = int(new_k_m / self.numMixtures)
        m_new = int(new_k_m % self.numMixtures)
        tw_new = tSen[k_t_new]
        
        newCurCentroid = self.centroids[tw_new][m_new] * self.numMembers[tw_new][m_new] + np.asarray(new_frame)
        self.numMembers[tw_new][m_new] += 1
        self.centroids[tw_new][m_new] = newCurCentroid / self.numMembers[tw_new][m_new]
       
      self.fCorpus[sentId] = newSent
      self.assignments[sentId] = newAssigns

    def align(self, fSen, tSen):
      fLen = fSen.shape[0]
      tLen = len(tSen)
      alignment = [0]*fLen
      align_probs = np.zeros((fLen, tLen))
      for i_f in range(fLen):
        dists = np.zeros((tLen, self.numMixtures))
        for i_t, tw in enumerate(tSen):
          for m in range(self.numMixtures):
            dists[i_t, m] = self.computeDist(fSen[i_f], self.centroids[tw][m])
        alignment[i_f] = int(np.argmin(np.min(dists, axis=1)))
        align_probs[i_f] = -np.min(dists, axis=1)   
      return alignment, align_probs.tolist()

    def printAlignment(self, filePrefix):
      f = open(filePrefix+'.txt', 'w')
      aligns = []
      for i, (fSen, tSen) in enumerate(zip(self.fCorpus, self.tCorpus)):
        alignment, align_probs = self.align(fSen, tSen)
        align_info = {
          'index': self.data_ids[i],
          'image_concepts': tSen,
          'alignment': alignment,
          'align_probs': align_probs,
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

def concatContext(feat, contextWidth):
    if contextWidth == 0:
      return feat
    else:
      nFrames, featDim = feat.shape
      nContext = 2 * contextWidth + 1
      pad = np.zeros((contextWidth, featDim))
      featPad = np.concatenate([pad, feat, pad]) 
      featStack = np.zeros((nFrames, featDim * nContext))
      for i in range(nFrames):
        featStack[i] = featPad[i:i+nContext].flatten(order=ORD)

      return featStack

if __name__ == "__main__":
  datapath = "./"
  mkmeans = KMeansWordDiscoverer(1, datapath + "small.npz", datapath + "small.txt")
  mkmeans.trainUsingEM(maxIterations=100, writeModel=True, modelPrefix="random_")
  mkmeans.printAlignment("random_pred")
