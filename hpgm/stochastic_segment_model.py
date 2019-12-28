import numpy as np
import json
import math
from scipy.stats import multivariate_normal
import time

class StochasticSegmentModel:
  def __init__(self, afeats, phn_labels, model_configs, model_name='stochastic_segment_model', debug=True):
    self.model_name = model_name
    self.feat_dim = model_configs.get('feature_dim', 14)
    self.cov_type = model_configs.get('cov_type', 'full')
    self.tol = model_configs.get('tol', 0.01)
     
    self.afeats = [[afeat.reshape(-1, self.feat_dim) for afeat in a_sent] for a_sent in afeats]
    self.phn_labels = phn_labels
    self.nframes = self.afeats[0][0].shape[0]
    if debug:
      print('self.afeats[0].shape: ', self.afeats[0][0].shape[0])
      print('self.nframes: ', self.nframes)
    self.compute_phone_distribution()
    if self.cov_type == 'full':
      self.obs_models = {'weights': np.ones((self.n_types - len(self.rare_phones)+1,)),
                       'means': np.zeros((self.n_types - len(self.rare_phones)+1, self.nframes, self.feat_dim)),
                       'covariances': np.zeros((self.n_types - len(self.rare_phones)+1, self.nframes, self.feat_dim, self.feat_dim)) 
                      } 
      for k in range(self.n_types - len(self.rare_phones)+1):
        for m in range(self.nframes):
          self.obs_models['covariances'][k, m] = self.tol*np.eye(self.feat_dim)

    elif self.cov_type == 'diag':
      self.obs_models = {'weights': np.ones((self.n_types - len(self.rare_phones)+1,)),
                       'means': np.zeros((self.n_types - len(self.rare_phones)+1, self.nframes, self.feat_dim)),
                       'covariances': np.zeros((self.n_types - len(self.rare_phones)+1, self.nframes, self.feat_dim,)) 
                      } 
    else:
      raise NotImplementedError

  def compute_phone_distribution(self):
    self.phn_ids = {}
    self.phn_counts = {}
    self.phns = []
    n_types = 0 
    for a_sent, phn_sent in zip(self.afeats, self.phn_labels):
      for afeat, phn in zip(a_sent, phn_sent):
        if phn not in self.phn_ids:
          self.phn_ids[phn] = n_types
          self.phns.append(phn)
          n_types += 1
        if phn not in self.phn_counts:
          self.phn_counts[phn] = 1
        else:
          self.phn_counts[phn] += 1

    self.n_types = n_types
    self.rare_phones = []
  
    for phn in self.phn_ids:
      if self.phn_counts[phn] < self.feat_dim:
        self.rare_phones.append(phn) 
        
    print('------ Training set summary ------')
    print('Num. of phone types: ', n_types)
    print('Num. of phones: ', sum(self.phn_counts.values()))
    print('Num. of sentences: ', len(self.phn_labels))
    print('Num. of phone types with members less than the embedding dimension: ', len(self.rare_phones))

  def train(self, debug=False):
    for ex, (a_sent, phn_sent) in enumerate(zip(self.afeats, self.phn_labels)):
      for afeat, phn in zip(a_sent, phn_sent):
        phn_id = self.phn_ids[phn]
        if phn in self.rare_phones:
          self.obs_models['means'][-1] += 1. / self.phn_counts[phn] * afeat
        else:
          if debug:
            print('afeat.shape: ', afeat.shape)
            print('means.shape: ', self.obs_models['means'][phn_id].shape)
          self.obs_models['means'][phn_id] += 1. / self.phn_counts[phn] * afeat
        
    for ex, (a_sent, phn_sent) in enumerate(zip(self.afeats, self.phn_labels)):
      begin_time = time.time()
      for afeat, phn in zip(a_sent, phn_sent):  
        phn_id = self.phn_ids[phn]

        if phn in self.rare_phones:
          for m in range(self.nframes):
            if self.cov_type == 'full':
              self.obs_models['covariances'][-1, m] += 1. / self.phn_counts[phn] * (afeat[m, :, np.newaxis] - self.obs_models['means'][-1, m, :, np.newaxis]) @ (afeat[m, np.newaxis, :] - self.obs_models['means'][-1, m])      
            elif self.cov_type == 'diag':
              self.obs_models['covariances'][-1, m] += 1. / self.phn_counts[phn] * (afeat[m, :] - self.obs_models['means'][-1, m, :]) ** 2
        else:
          for m in range(self.nframes):
            if debug:
              print('self.obs_models.shape: ', self.obs_models['covariances'].shape)
            if self.cov_type == 'full':
              self.obs_models['covariances'][phn_id, m] += 1. / self.phn_counts[phn] * (afeat[m, :, np.newaxis] - self.obs_models['means'][phn_id, m, :, np.newaxis]) @ (afeat[m, np.newaxis, :] - self.obs_models['means'][phn_id, m])
            elif self.cov_type == 'diag':
              self.obs_models['covariances'][phn_id, m] += 1. / self.phn_counts[phn] * (afeat[m, :] - self.obs_models['means'][phn_id, m, :]) ** 2
            else:
              raise NotImplementedError
      print('Take %.3f to process example %d' % (time.time()-begin_time, ex))

    np.save(self.model_name+'_means', self.obs_models['means'])
    np.save(self.model_name+'_covariances', self.obs_models['covariances'])
    with open(self.model_name+'_phones.txt', 'w') as f:
      f.write('\n'.join(self.phns))
    
    self.predict(self.afeats, debug=debug)

  def predict(self, afeats, debug=False):
    predict_phn_sents = []
    for ex, a_sent in enumerate(afeats):
      begin_time = time.time()
      predict_phn_sent = [] 
      for afeat in a_sent:
        scores = -np.inf*np.ones((self.n_types,))
        if self.cov_type == 'full':
          for k in range(self.n_types):
            scores[k] = 0.
            if self.phns[k] not in self.rare_phones:
              for m in range(self.nframes):
                if np.linalg.det(self.obs_models['covariances'][k, m]) <= 0:
                  print('Bad covariance: ', self.obs_models['covariances'][k, m]) 
                  print(np.linalg.det(self.obs_models['covariances'][k, m]))
                scores[k] += gaussian(afeat[m], self.obs_models['means'][k, m], self.obs_models['covariances'][k, m], log_prob=True, cov_type=self.cov_type)  
            else:
              for m in range(self.nframes):
                scores[k] += gaussian(afeat[m], self.obs_models['means'][-1, m], self.obs_models['covariances'][-1, m], log_prob=True, cov_type=self.cov_type)  
          predict_phn_sent.append(self.phns[np.argmax(scores)])
        elif self.cov_type == 'diag':
          for k in range(self.n_types):
            scores[k] = 0.
            if self.phns[k] not in self.rare_phones:
              scores[k] += np.sum(gaussian(afeat, self.obs_models['means'][k], self.obs_models['covariances'][k], log_prob=True, cov_type=self.cov_type)) 
            else:
              scores[k] += np.sum(gaussian(afeat, self.obs_models['means'][-1], self.obs_models['covariances'][-1], log_prob=True, cov_type=self.cov_type)) 

          predict_phn_sent.append(self.phns[np.argmax(scores)])

      
      predict_phn_sents.append(' '.join(predict_phn_sent))
      print('Takes %.3f to predict example %d' % (time.time()-begin_time, ex))

    with open(self.model_name+'_predicted_phones.txt', 'w') as f:
      f.write('\n'.join(predict_phn_sents))

def gaussian(x, mean, cov, cov_type='full', log_prob=True):
  d = mean.shape[0]
  prob = None
  if cov_type == 'diag':
    assert np.min(cov) > 0.
    if log_prob:
      log_norm_const = float(d) / 2. * np.log(2. * math.pi) + np.sum(np.log(cov)) / 2. 
      prob = - log_norm_const - np.sum((x - mean) ** 2 / (2. * cov), axis=-1) 
    else:
      norm_const = np.sqrt(2. * math.pi) ** float(d)
      norm_const *= np.prod(np.sqrt(cov)) 
      x_z = x - mean
      prob = np.exp(- np.sum(x_z ** 2 / (2. * cov), axis=-1)) / norm_const 
  elif cov_type == 'full':
    if log_prob:
      assert np.linalg.det(cov) > 0.
      norm_const = np.log(2. * math.pi) * float(d) / 2
      norm_const += np.log(np.linalg.det(cov)) / 2
      x_zm = x[np.newaxis, :] - mean
      prob = - x_zm @ np.linalg.inv(cov) @ x_zm.T / 2. -  norm_const
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError
  return prob

if __name__ == '__main__':
  tasks = [0, 1]
  #feature_file = '../data/mscoco/mscoco_kamper_embeddings_phone_power_law_cmvn.npz'
  #phone_label_file = '../data/mscoco/mscoco_subset_subword_level_power_law_phones.json'
  feature_file = '../data/TIMIT/TIMIT_subset_kamper_embeddings.npz'
  phone_label_file = '../data/TIMIT/TIMIT_subset_phones.json'
  
  if 0 in tasks:
    print('Test utilities to compute the gaussian log prob ...')
    mean = np.random.normal(size=(2,))
    cov = np.array([[0.5, 0.], [0., 0.5]])
    a = np.random.normal(size=(2,))
    print('scipy gaussian: ', np.log(multivariate_normal.pdf(a, mean, cov)))
    print('My gaussian: ', gaussian(a, mean, np.diag(cov), cov_type='diag'))
  if 1 in tasks:
    exp_dir = 'exp/dec3_seg_TIMIT/'
    model_configs = {}
    afeats_npz = np.load(feature_file)
    afeats = [afeats_npz[k] for k in sorted(afeats_npz, key=lambda x:int(x.split('_')[-1]))]
    with open(phone_label_file, 'r') as f:
      phn_dict = json.load(f)
    phn_labels = [phn_dict[k] for k in sorted(afeats_npz, key=lambda x:int(x.split('_')[-1]))]
    
    model = StochasticSegmentModel(afeats, phn_labels, model_configs, exp_dir+'stochastic_segment', debug=False)
    model.train()
  if 2 in tasks:
    exp_dir = 'exp/dec3_seg_diag_TIMIT/'
    model_configs = {'cov_type': 'diag'}
    afeats_npz = np.load(feature_file)
    afeats = [afeats_npz[k] for k in sorted(afeats_npz, key=lambda x:int(x.split('_')[-1]))]
    with open(phone_label_file, 'r') as f:
      phn_dict = json.load(f)
    phn_labels = [phn_dict[k] for k in sorted(afeats_npz, key=lambda x:int(x.split('_')[-1]))]
    
    model = StochasticSegmentModel(afeats, phn_labels, model_configs, exp_dir+'stochastic_segment', debug=False)
    model.train(debug=True)
