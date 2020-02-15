import numpy as np
import math
import time
from copy import deepcopy
from scipy.misc import logsumexp
from scipy.misc import imresize 
from scipy.special import digamma
from scipy.special import kl_div
from scipy import signal
from sklearn.cluster import KMeans
import json
from PIL import Image

ORDER='C'
class ImagePhoneMixtureWordDiscoverer:
  def __init__(self, speech_feature_file, image_feature_file, model_configs, model_name='image_phone_mixture_word_discoverer'):
    self.model_name = model_name
    self.a_corpus = []
    self.v_corpus = []

    # Prior parameters
    self.Kmax = model_configs.get('Kmax', 100)
    self.Mmax = model_configs.get('Mmax', 3)
   
    g_sb00 = model_configs.get('gamma_stick_break', 1.)
    self.g_sb0 = -np.inf * np.ones((self.Kmax, 2))
    for k in range(self.Kmax):
      self.g_sb0[k, 0] = np.log(g_sb00)
      if k < self.Kmax - 1:
        self.g_sb0[k, 1] = np.log(g_sb00 * (self.Kmax - k - 1))
         
    self.g_a0 = model_configs.get('gamma_align', 0.01)
    self.g_a0 = np.log(self.g_a0)
    self.g_mv0 = model_configs.get('gamma_visual_mixture', 0.01)
    self.g_t0 = model_configs.get('gamma_trans_prob', 0.01)
    self.g_t0 = np.log(self.g_t0)
    self.k_v0 = model_configs.get('k_visual', 0.01) # Ratio between obs variance and prior variance
    self.k_v0 = np.log(self.k_v0)

    self.has_null = model_configs.get('has_null', True)
    #self.fixed_variance = model_configs.get('fixed_variance', False)
    self.read_corpus(speech_feature_file, image_feature_file)

    self.len_probs = {}
    self.concept_prior = None 
    self.align_init = {}
    self.audio_obs_model = {'trans_probs': None}
    self.image_obs_model = {'weights': None,
                            'means': None,
                            'n_mixures': None
                            }

  def read_corpus(self, speech_feat_file, image_feat_file):
    a_corpus_str = []
    self.phone2idx = {}
    n_types = 0
    n_phones = 0
    
    f = open(speech_feat_file, 'r')
    for line in f:
      a_sent = line.strip().split()
      a_corpus_str.append(a_sent)
      for phn in a_sent:
        if phn not in self.phone2idx:
          self.phone2idx[phn] = n_types
          n_types += 1
        n_phones += 1
    f.close()
    self.audio_feat_dim = n_types
    
    self.a_corpus = []
    # XXX
    for a_sent_str in a_corpus_str:
      a_sent = np.zeros((len(a_sent_str), n_types))
      for t, phn in enumerate(a_sent_str):
        a_sent[t, self.phone2idx[phn]] = 1.
      self.a_corpus.append(a_sent)

    v_npz = np.load(image_feat_file)
    # XXX
    self.v_corpus = [np.array(v_npz[k]) for k in sorted(v_npz, key=lambda x:int(x.split('_')[-1]))] 
    self.image_feat_dim = self.v_corpus[0].shape[-1]
    n_images = 0
    for ex, vfeat in enumerate(self.v_corpus):
      n_images += len(vfeat)
      if vfeat.shape[-1] == 0:
        print('ex: ', ex)
        print('vfeat empty: ', vfeat.shape) 
        self.v_corpus[ex] = np.zeros((1, self.image_feat_dim))
     
    if self.has_null:
      self.v_corpus = [np.concatenate((np.zeros((1, self.image_feat_dim)), vfeat), axis=0) for vfeat in self.v_corpus]  
    
    # XXX
    #self.a_corpus = self.a_corpus[:10]
    #self.v_corpus = self.v_corpus[:10]
    print(len(self.v_corpus), len(self.a_corpus))
    assert len(self.v_corpus) == len(self.a_corpus)

    print('----- Corpus Summary -----')
    print('Number of examples: ', len(self.a_corpus))
    print('Number of phonetic categories: ', n_types)
    print('Number of phones: ', n_phones)
    print('Number of objects: ', n_images)

  def initialize_model(self, alignments=None, method='kmeans'):
    begin_time = time.time()
    self.compute_translation_length_probabilities()

    for lv in self.len_probs:
      self.align_init[lv] = -np.log(lv) * np.ones((lv,))

    self.vs = np.zeros((self.Kmax,)) 
    for k in range(self.Kmax):
      self.vs[k] = np.exp(self.g_sb0[k, 0]) / (np.exp(self.g_sb0[k, 0]) + np.exp(self.g_sb0[k, 1]))
    self.concept_prior = np.log(compute_stick_break_prior(self.vs)) 
    
    self.audio_obs_model['trans_probs'] = 1./self.audio_feat_dim * np.ones((self.Kmax, self.audio_feat_dim))

    self.mu_v0 = np.zeros((self.image_feat_dim,))
    n_frames_v = 0.
    for vfeat in self.v_corpus:
      #print('afeat.shape, vfeat.shape: ', afeat.shape, vfeat.shape) 
      self.mu_v0 += np.sum(vfeat, axis=0)
      n_frames_v += vfeat.shape[0]
    self.mu_v0 /= n_frames_v
    
    c = 3.    
    self.fixed_variance_v = 0. # XXX: Assume fixed variances  
    n_corpus = len(self.v_corpus)    
    if method == 'rand':
      for k in range(self.Kmax):
        for m in range(self.Mmax):
          #i_a = np.random.randint(0, n_frames_a-1)
          #i_v = np.random.randint(0, n_frames_v-1)
          #cluster_centers_a[k, m] = a_corpus_concat[i_a]
          #cluster_centers_v[k, m] = v_corpus_concat[i_v] 
          i_ex = np.random.randint(0, n_corpus-1)
          if len(self.v_corpus[i_ex]) <= 1:
            i_v = 0
          else:
            i_v = np.random.randint(0, len(self.v_corpus[i_ex])-1)
          cluster_centers_v[k, m] = self.v_corpus[i_ex][i_v]
    else:
      cluster_centers_v = KMeans(n_clusters=self.Mmax * self.Kmax, max_iter=10).fit(np.concatenate(self.v_corpus, axis=0)).cluster_centers_.reshape(self.Kmax, self.Mmax, -1)

    for vfeat in self.v_corpus: 
      self.fixed_variance_v += np.sum((vfeat - self.mu_v0)**2) / (c * n_frames_v * self.image_feat_dim) 
        
    self.image_obs_model['means'] = cluster_centers_v
    self.image_obs_model['weights'] = np.log(1./self.Mmax) * np.ones((self.Kmax, self.Mmax))
    self.image_obs_model['n_mixtures'] = self.Mmax * np.ones((self.Kmax,)) 
    self.image_obs_model['variances'] = self.fixed_variance_v * np.ones((self.Kmax, self.Mmax, self.image_feat_dim))       
      
    # Initialize hyperparameters for the approximate parameter posteriors
    self.g_aN = {l: (self.g_a0 - np.log(l)) * np.ones((l,)) for l in self.len_probs}  # size-L dict of Nv-d array
    self.g_sbN = deepcopy(self.g_sb0) # K x 2 matrix
    self.g_tN = self.g_t0 * np.ones((self.Kmax,)) # K-d array
    self.g_mvN = self.g_mv0 * np.ones((self.Kmax, self.Mmax)) # Mv-d array 
    self.k_vN = self.k_v0 * np.ones((self.Kmax, self.Mmax)) 
    self.update_digamma_functions()

    # Initialize the approximate posteriors
    self.counts_concept = None #[np.array([deepcopy(self.concept_prior) for _ in range(vfeat.shape[0])]) for vfeat in self.v_corpus] 
    self.counts_image_mixture = None 
    self.counts_align = [np.log(1./len(vfeat)) * np.ones((len(afeat), len(vfeat))) for afeat, vfeat in zip(self.a_corpus, self.v_corpus)] 
         
  def train_using_EM(self, num_iterations=10, write_model=True):
    self.initialize_model()
    #print('initial audio_obs_means: ', self.audio_obs_model['means'])
    #print('initial image_obs_means: ', self.image_obs_model['means'])
    #print('initial self.concept_prior: ', np.exp(self.concept_prior))
    if write_model:
      self.print_model(self.model_name+'_initial_model.txt') 
    
    begin_time = time.time()
    print("Initial log likelihood: ", self.compute_log_likelihood()) 
    print("Take %.5f s to compute log-likelihood" % (time.time()-begin_time))
    for n in range(num_iterations):
      begin_time = time.time()
      new_counts_concept = [] # length-Nd list of Nv x K x Ma x Mv array
      new_counts_align = [] 
      new_counts_image_mixture = []
      
      for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
        log_probs_a_given_z = self.log_probs_afeat_given_z(afeats)
        log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)

        # Estimate counts
        counts_concept = self.update_concept_counts(self.counts_align[ex], log_probs_a_given_z, log_probs_v_given_zm)
        new_counts_concept.append(counts_concept)
       
        counts_image_mixture = self.update_image_mixture_counts(log_probs_v_given_zm) 
        new_counts_image_mixture.append(counts_image_mixture)

        counts_align = self.update_alignment_counts(log_probs_a_given_z, counts_concept) 
        new_counts_align.append(counts_align)
           
      print('Take %.5f s for E step' % (time.time() - begin_time)) 
      begin_time = time.time()
      self.counts_concept = deepcopy(new_counts_concept)
      self.counts_image_mixture = deepcopy(new_counts_image_mixture)
      self.counts_align = deepcopy(new_counts_align)
      # Update posterior hyperparameters
      self.update_posterior_hyperparameters()
      self.update_digamma_functions()

      # Update parameters
      self.update_concept_alignment_model()
      self.update_audio_observation_models() 
      self.update_image_observation_models()
      print('Take %.5f s for M step' % (time.time() - begin_time))  
      
      begin_time = time.time()
      print('Log likelihood after iteration %d: %.5f' % (n, self.compute_log_likelihood()))
      print("Take %.5f s to compute log-likelihood" % (time.time()-begin_time))
      if write_model:
        self.print_model(self.model_name+'_model_iter_%d' % n) 

      #print('ELBO after iteration %d: %.5f' % (n, self.compute_ELBO()))
      self.print_alignment(self.model_name)
    print('Final Log likelihood after iteration %d: %.5f' % (n, self.compute_log_likelihood()))

  # Compute log likelihood using the formula: \log p(a, v) = \sum_i=1^n \log p(v_i) + \log \sum_{A} p(A|v) * p(x|A, v)
  def compute_log_likelihood(self):
    ll = 0.
    for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
      #print('vfeats.shape, afeats.shape: ', vfeats.shape, afeats.shape)
      #print('self.obs_audio_model means', self.audio_obs_model['means'].shape)
      #print('self.obs_visual_model means', self.image_obs_model['means'].shape)
 
      log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)
      log_probs_a_given_z = self.log_probs_afeat_given_z(afeats)
      log_probs_v_given_z = logsumexp(log_probs_v_given_zm + self.image_obs_model['weights'], axis=-1)
      log_probs_v = logsumexp(self.concept_prior + log_probs_v_given_z, axis=1)
      # TODO: Check sum = 1  
      log_probs_z_given_v = ((self.concept_prior + log_probs_v_given_z).T - log_probs_v).T
      #if (np.max(log_probs_z_given_v) - np.min(log_probs_z_given_v)) > 10000:
      #  print('self.image_obs_model weights: ', max(self.image_obs_model['weights']), min(self.image_obs_model['weights']))
      #  print('concept_prior range: ', max(self.concept_prior), min(self.concept_prior))
      #print('sum(p(z)): ', np.sum(np.exp(self.concept_prior)))
      #print('sum(p(z|v)): ', np.sum(np.exp(log_probs_z_given_v), axis=1))
      log_prob_v_tot = np.sum(log_probs_v)
      log_probs_a_i_given_v = self.log_prob_afeats_align_given_vfeats(log_probs_a_given_z, log_probs_z_given_v)
      #print(log_probs_a_i_given_v.shape)
      log_prob_a_given_v_tot = logsumexp(log_probs_a_i_given_v.flatten())
      ll += 1. / len(self.a_corpus) * (log_prob_v_tot + log_prob_a_given_v_tot)
    return ll
  
  def update_concept_counts(self, counts_align_init, probs_afeat_given_z, probs_vfeat_given_zm):
    T = len(probs_afeat_given_z)
    n_state = len(probs_vfeat_given_zm)
    counts_concept = -np.inf * np.ones((n_state, self.Kmax))
    
    for i in range(n_state):
      counts_concept[i] = deepcopy(self.digammas_concept)
    
    counts_concept += np.exp(counts_align_init).T @ probs_afeat_given_z 
    counts_concept += np.sum(self.digammas_mv + probs_vfeat_given_zm, axis=-1)
 
    # Normalize
    for i in range(n_state):
      counts_concept[i] -= logsumexp(counts_concept[i])
    
    return counts_concept

  def update_image_mixture_counts(self, probs_vfeat_given_zm):
    counts_image_mixture = self.digammas_mv + probs_vfeat_given_zm
    # Normalize
    norm_factors = logsumexp(counts_image_mixture, axis=-1) 
    counts_image_mixture = np.transpose(np.transpose(counts_image_mixture, (2, 0, 1)) - norm_factors, (1, 2, 0)) 
    return counts_image_mixture 
   
  def update_alignment_counts(self, log_probs_a_given_z, counts_concept):
    n_states = counts_concept.shape[0]
    counts_align = self.log_probs_afeat_given_i(log_probs_a_given_z, counts_concept)
    counts_align += self.digammas_align[n_states]
    counts_align = (counts_align.T - logsumexp(counts_align, axis=-1)).T

    assert not np.isnan(counts_align).any()
    return counts_align 
  
  def update_posterior_hyperparameters(self):
    new_g_aN = {l: (self.g_a0 - np.log(l)) * np.ones((l,)) for l in self.len_probs}
    new_g_mvN = (self.g_mv0 - np.log(self.Mmax)) * np.ones((self.Kmax, self.Mmax))
    new_g_sbN = deepcopy(self.g_sb0) 
    new_k_vN = self.k_v0 * np.ones((self.Kmax, self.Mmax))
    
    aggregated_counts = None
    aggregated_counts = [logsumexp(counts, axis=0) for counts in self.counts_image_mixture]
    new_g_mvN = logsumexp(np.array([new_g_mvN]+aggregated_counts), axis=0) 
  
    aggregated_counts_init = {n_states:[] for n_states in self.len_probs}
    for counts_init in self.counts_align:
      n_states = counts_init.shape[1]
      aggregated_counts_init[n_states].append(logsumexp(counts_init, axis=0)) 

    for n_states in self.len_probs:    
      new_g_aN[n_states] = logsumexp(np.array([new_g_aN[n_states]]+aggregated_counts_init[n_states]), axis=0)
          
    aggregated_counts, aggregated_counts_gt = [], []
    for counts in self.counts_concept:
      n_states = counts.shape[0]
      counts_gt = -np.inf * np.ones((n_states, self.Kmax))
      for k in range(self.Kmax-1):
        counts_gt[:, k] = logsumexp(counts[:, k+1:], axis=-1) 
      aggregated_counts.append(logsumexp(counts, axis=0)) 
      aggregated_counts_gt.append(logsumexp(counts_gt, axis=0))
    new_g_sbN[:, 0] = logsumexp(np.array([new_g_sbN[:, 0]]+aggregated_counts), axis=0)
    new_g_sbN[:, 1] = logsumexp(np.array([new_g_sbN[:, 1]]+aggregated_counts_gt), axis=0)

    self.g_mvN = new_g_mvN
    self.g_aN = new_g_aN
    self.g_sbN = new_g_sbN

  def update_digamma_functions(self):
    self.digammas_align = {l:None for l in self.g_aN} 
    
    for l in self.g_aN:
      self.digammas_align[l] = digamma(np.exp(self.g_aN[l])) - digamma(np.exp(logsumexp(self.g_aN[l])))
    self.digammas_concept = np.zeros((self.Kmax,))
    self.digammas_mv = np.zeros((self.Kmax, self.Mmax))
    
    for k in range(self.Kmax):
      for j in range(k-1):
        self.digammas_concept[k] += digamma(np.exp(self.g_sbN[j, 1])) - digamma(np.exp(logsumexp(self.g_sbN[j])))
      self.digammas_concept[k] += digamma(np.exp(self.g_sbN[k, 0])) - digamma(np.exp(logsumexp(self.g_sbN[k])))
      for m in range(self.Mmax):
        self.digammas_mv[k] = digamma(np.exp(self.g_mvN[k])) - digamma(np.exp(logsumexp(self.g_mvN[k])))
  
  def update_concept_alignment_model(self, ):
    vs = np.exp(self.g_sbN[:, 0]) / (np.exp(self.g_sbN[:, 0]) + np.exp(self.g_sbN[:, 1]))
    
    self.concept_prior = None
    self.concept_prior = np.log(compute_stick_break_prior(vs))
    
    for l in self.g_aN:
      self.align_init[l] = self.g_aN[l] - logsumexp(self.g_aN[l])
 
  def update_audio_observation_models(self):
    self.audio_obs_model['trans_probs'] = np.exp(self.g_t0) * np.ones((self.Kmax, self.audio_feat_dim))
    for ex, (afeat, vfeat) in enumerate(zip(self.a_corpus, self.v_corpus)):
      T = len(afeat)
      counts_z_a = np.zeros((T, self.Kmax))
      for t in range(T):
        counts_z_a[t] = logsumexp(self.counts_concept[ex].T + self.counts_align[ex][t], axis=-1)
        if np.isnan(counts_z_a[t]).any():
          print('counts_z_a_t: ', self.counts_concept[ex].T + self.counts_align[ex][t])
      self.audio_obs_model['trans_probs'] += np.exp(counts_z_a).T @ afeat
    self.audio_obs_model['trans_probs'] = (self.audio_obs_model['trans_probs'].T / np.sum(self.audio_obs_model['trans_probs'], axis=1)).T      
  
  def update_image_observation_models(self):
    self.image_obs_model['weights'] = -np.inf * np.ones(self.g_mvN.shape)
    exp_image_mixture_means = np.tile(np.exp(self.k_v0) * self.mu_v0, (self.image_obs_model['means'].shape[0], self.image_obs_model['means'].shape[1], 1))
    exp_num_image_mixture = np.zeros((self.Kmax, self.Mmax))
    
    for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
      for k in range(self.Kmax):
        counts_mv_z = (self.counts_concept[ex][:, k] + self.counts_image_mixture[ex][:, k, :].T)
        exp_image_mixture_means[k] += np.exp(counts_mv_z) @ vfeats
        exp_num_image_mixture[k] += np.exp(logsumexp(counts_mv_z, axis=1))
    
    self.image_obs_model['means'] = np.transpose(np.transpose(exp_image_mixture_means, (2, 0, 1)) / (np.exp(self.k_v0) + exp_num_image_mixture), (1, 2, 0))
    self.image_obs_model['weights'] = (self.g_mvN.T - logsumexp(self.g_mvN, axis=1)).T         

  def compute_translation_length_probabilities(self):  
    for afeats, vfeats in zip(self.a_corpus, self.v_corpus):
      if len(vfeats) not in self.len_probs:
        self.len_probs[len(vfeats)] = {}
      if len(afeats) not in self.len_probs[len(vfeats)]:
        self.len_probs[len(vfeats)][len(afeats)] = 1
      else:
        self.len_probs[len(vfeats)][len(afeats)] += 1

    for l in self.len_probs:
      tot_count = sum(self.len_probs[l].values())
      for m in self.len_probs[l]:
        self.len_probs[l][m] /= tot_count 
   
  def log_prob_afeats_align_given_vfeats(self, log_probs_a_given_z, log_probs_z_given_v):
    T = len(log_probs_a_given_z)
    n_states = len(log_probs_z_given_v)
    log_probs_a_i_given_v = -np.inf * np.ones((T, n_states)) 
    for i in range(n_states):
      log_probs_a_i_given_v[:, i] = self.align_init[n_states][i] + logsumexp(log_probs_z_given_v[i] + log_probs_a_given_z, axis=-1)
      #if np.isnan(log_probs_a_i_given_v[:, i]).any():
      #  print('log_probs_a_given_z range: ', np.max(log_probs_a_given_z), np.min(log_probs_a_given_z))
      #  print('align_init: ', self.align_init[n_states])
      #  print('log_probs_z_given_v[i] range: ', max(log_probs_z_given_v[i]), min(log_probs_z_given_v[i])) 
    return log_probs_a_i_given_v
  
  def log_probs_afeat_given_i(self, log_probs_a_given_z, counts_concept):
    return log_probs_a_given_z  @ np.exp(counts_concept).T   
    
  def log_probs_afeat_given_z(self, afeat):
    return np.log(afeat @ self.audio_obs_model['trans_probs'].T)

  def log_probs_vfeat_given_z_m(self, vfeats):
    n_states = vfeats.shape[0]
    log_probs = []
    
    for m in range(self.Mmax):   
      log_probs_m = []
      for i in range(n_states):
        log_probs_m.append(compute_diagonal_gaussian(vfeats[i], self.image_obs_model['means'][:, m], self.image_obs_model['variances'][:, m, :], log_prob=True))
      log_probs.append(log_probs_m)

    return np.transpose(np.asarray(log_probs), (1, 2, 0))

  def align_i(self, i): 
    T = self.a_corpus[i].shape[0]
    n_states = self.v_corpus[i].shape[0]
     
    concept_scores = self.counts_concept[i]
    align_scores = self.counts_align[i]
    best_alignment = np.argmax(align_scores, axis=1).tolist()
    best_concepts = np.argmax(concept_scores, axis=1).tolist()
    
    return best_concepts, concept_scores.tolist(), best_alignment, align_scores.tolist()

  def print_model(self, filename):
    init_file = open(filename+'_alignment_initialprobs.txt', 'w')
    for n_state in sorted(self.len_probs):
      for i in range(n_state):
        init_file.write('%d\t%d\t%f\n' % (n_state, i, self.align_init[n_state][i]))
    init_file.close()

    obs_a_file = filename+'_audio_mixture_parameters.npz'
    obs_v_file = filename+'_image_mixture_parameters.npz'
    concept_prob_file = filename+'_concept_probabilities.npz'

    np.savez(obs_a_file, **self.audio_obs_model)
    np.savez(obs_v_file, **self.image_obs_model)
    np.savez(concept_prob_file, self.concept_prior)
    
    with open(filename+'_phone2idx.json', 'w') as f:
      json.dump(self.phone2idx, f)

  def print_alignment(self, file_prefix):
    f = open(file_prefix+'.txt', 'w')
    alignments = []
    n_data = len(self.a_corpus)
    for ex in range(n_data):
      concepts, concept_probs, alignment, align_probs = self.align_i(ex)
      #concepts = None
      #alignment, align_probs = self.align(afeat, vfeat)
      align_info = {
          'index': ex,
          'image_concepts': concepts, 
          'alignment': alignment,
          'align_probs': align_probs
        }
      alignments.append(align_info)
      for a in alignment:
        f.write('%d\t' % a)
      f.write('\n\n')
    f.close()

    with open(file_prefix+'.json', 'w') as f:
      json.dump(alignments, f, indent=4, sort_keys=True)

def compute_stick_break_prior(vs):
  K = len(vs)
  pvs = np.cumprod(1-vs)
  prior = np.zeros((K,))
  prior[0] = vs[0]
  prior[1:] = pvs[:-1] * vs[1:]
  return prior

def compute_diagonal_gaussian(x, mean, cov, log_prob=False):
  d = mean.shape[-1]
  assert np.min(cov) > 0.
  if log_prob:
    log_norm_const = float(d) / 2. * np.log(2. * math.pi) + np.sum(np.log(cov), axis=-1) / 2.
    prob = - log_norm_const - np.sum((x - mean) ** 2 / (2. * cov), axis=-1)
  else:
    norm_const = np.sqrt(2. * math.pi) ** float(d)
    norm_const *= np.prod(np.sqrt(cov), axis=-1)
    x_z = x - mean
    prob = np.exp(- np.sum(x_z ** 2 / (2. * cov), axis=-1)) / norm_const
  return prob

def gaussian(x, mean, cov, cov_type='diag', log_prob=False):
  d = mean.shape[0]
  if cov_type == 'diag':
    assert np.min(np.diag(cov)) > 0.
    if log_prob:
      log_norm_const = float(d) / 2. * np.log(2. * math.pi) + np.sum(np.log(np.diag(cov))) / 2.
      prob = - log_norm_const - np.sum((x - mean) ** 2 / (2. * np.diag(cov)), axis=-1) 
    else:
      norm_const = np.sqrt(2. * math.pi) ** float(d)
      norm_const *= np.prod(np.sqrt(np.diag(cov))) 
      x_z = x - mean
      prob = np.exp(- np.sum(x_z ** 2 / (2. * np.diag(cov)), axis=-1)) / norm_const 
  else:
    assert np.linalg.det(cov) > 0.
    chol_cov = np.linalg.cholesky(cov)
    norm_const = np.sqrt(2. * math.pi) ** float(d)
    norm_const *= np.linalg.det(chol_cov)
    prob = np.exp(-np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), x - mean) / 2.) / norm_const

  return prob

# TODO
def KL_divergence(p, q, log_prob=True):  
  if log_prob:
    return np.sum(np.exp(p) * (p - q))
  else:
    is_zero_p = (p == 0.)
    return np.sum(p * (np.log((p + is_zero_p) / q)))

def gaussian_KL_divergence(mean1, cov1, mean2, cov2, cov_type='diag'):
  #print('mean and cov shapes: ', mean1.shape, cov1.shape, mean2.shape, cov2.shape)
  #print('mean and cov: ', mean1, cov1, mean2, cov2)
  assert np.min(np.diagonal(cov1, axis1=-1, axis2=-2)) > 0. and np.min(np.diagonal(cov2, axis1=-1, axis2=-2)) > 0.
  assert mean1.shape[-1] == cov1.shape[-1] and mean2.shape[-1] == cov2.shape[-1] and mean1.shape == mean2.shape

  if cov_type == 'diag':
    d = mean1.shape[-1]
    tr_cov2_inv_cov1 = np.sum(np.diagonal(cov1, axis1=-1, axis2=-2) / np.diagonal(cov2, axis1=-1, axis2=-2))
    mahalanobis = np.sum((mean2-mean1)**2 / np.diagonal(cov2, axis1=-1, axis2=-2), axis=-1)
    log_det_cov1_inv_cov2 = np.sum(np.log(np.diagonal(cov2, axis1=-1, axis2=-2) / np.diagonal(cov1, axis1=-1, axis2=-2)), axis=-1)
    return np.sum(1./2 * (tr_cov2_inv_cov1 + mahalanobis + log_det_cov1_inv_cov2 - d)) 
  if cov_type == 'standard_prior':
    return 1./2 * np.sum(np.diagonal(cov1, axis1=-1, axis2=-2) + mean1**2 - np.log(np.diagonal(cov1, axis1=-1, axis2=-2)) - 1.)      
  
if __name__ == '__main__':  
  test_case = 3

  if test_case == 0:
    exp_dir = 'exp/feb_5_tiny/'
    print(exp_dir)
    # Test KL-divergence
    p = np.array([0.1, 0.4, 0.5])
    q = np.array([0.2, 0.6, 0.2])
    print('my KL_divergence: ', KL_divergence(np.log(p), np.log(q)))
    print('scipy KL_divergence: ', np.sum(kl_div(p, q)))
  
    # Test on noisy one-hot vectors
    eps = 0.
    # ``2, 1``, ``3, 2``, ``3, 1``
    image_feats = {'0':np.array([[eps/2., 1.-eps, eps/2.], [1-eps, eps/2., eps/2.]]), '1':np.array([[eps/2., eps/2., 1.-eps], [eps/2., 1.-eps, eps/2.]]), '2':np.array([[eps/2., eps/2., 1.-eps], [1.-eps, eps/2., eps/2.]])}
    audio_feats = '0 1\n1 2\n2 0'
    with open(exp_dir + 'tiny.txt', 'w') as f:
      f.write(audio_feats)
    np.savez(exp_dir + 'tiny.npz', **image_feats)
  
    alignments = None
    model_configs = {'Kmax':3, 'Mmax':1, 'embedding_dim':3, 'has_null':False}
    speechFeatureFile = exp_dir + 'tiny.txt'
    imageFeatureFile = exp_dir + 'tiny.npz'
    model = ImagePhoneMixtureWordDiscoverer(speechFeatureFile, imageFeatureFile, model_configs=model_configs, model_name='tiny')
    model.train_using_EM(num_iterations=10)
    model.print_alignment(exp_dir + 'tiny')
  elif test_case == 1:  
    img = Image.open('../1000268201.jpg')
    width = 10
    height = 10
    img = imresize(np.array(img), size=(width, height)).reshape(width*height, 3)
    np.savez("img.npz", **{'arr_0':img})

    # Test on image pixels
    alignments = None
    #model_configs = {'Kmax':3, 'Mmax':1, 'embedding_dim':3, 'gamma_sb':10*np.ones((3, 2)), 'alignments':[alignments], 'has_null':False}  
    model_configs = {'Kmax':3, 'Mmax':1, 'embedding_dim':3, 'alignments':[alignments], 'has_null':False, 'initialize_method':'kmeans'}  
    
    speechFeatureFile = '../img_vec1.npz'
    imageFeatureFile = '../img_vec1.npz'
    model = ImagePhoneMixtureWordDiscoverer(speechFeatureFile, imageFeatureFile, model_configs=model_configs, model_name='image_audio')
    model.train_using_EM(num_iterations=1)
    model.print_alignment('image_segmentation')
  elif test_case == 2:   
    # Test on a single example with known alignment
    align_file = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json'
    segment_file = '../data/flickr30k/audio_level/flickr30k_gold_landmarks_mfcc.npz'
    speech_feature_file = '../data/flickr30k/sensory_level/flickr_concept_kamper_embeddings.npz'
    image_feature_file = '../data/flickr30k/sensory_level/flickr30k_vgg_penult.npz'
    exp_dir = 'exp/oct_28_flickr_mfcc/'
     
    npz_file = np.load(segment_file)
    segmentations = []
    for k in sorted(npz_file, key=lambda x:int(x.split('_')[-1]))[:1]:
      segmentations.append(npz_file[k])
    
    alignments = []
    with open(align_file, 'r') as f:
      align_dicts = json.load(f)
    alignments.append(align_dicts[0]['alignment'])
    
    segment_alignments = [] 
    for start, end in zip(segmentations[0][:-1], segmentations[0][1:]):    
      segment_alignments.append(alignments[0][start]) 
        
    print('segment_alignments: ', segment_alignments)
    
    model_configs = {'Kmax':65, 'Mmax':1, 'embedding_dim':120, 'alignments':[segment_alignments], 'segmentations':segmentations}  
     
    model = ImagePhoneMixtureWordDiscoverer(speech_feature_file, image_feature_file, model_configs=model_configs, model_name=exp_dir+'image_audio')
    model.train_using_EM(num_iterations=10)
  elif test_case == 3:
    model_configs = {'Kmax':65, 'Mmax':1, 'has_null':False}
    #align_file = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json'
    #segment_file = '../data/flickr30k/audio_level/flickr30k_gold_landmarks_mfcc.npz'
    #speech_feature_file = '../data/flickr30k/sensory_level/flickr_concept_kamper_embeddings.npz'
    #speech_feature_file = '../data/mscoco/src_mscoco_subset_subword_level_power_law.txt'
    # XXX
    speech_feature_file = '../data/mscoco/src_mscoco_subset_130k_power_law_phone_captions.txt'
    #image_feature_file = '../data/flickr30k/sensory_level/flickr30k_vgg_penult.npz'
    #image_feature_file = '../data/mscoco/mscoco_subset_subword_level_concept_gaussian_vectors.npz'
    image_feature_file = '../data/mscoco/mscoco_subset_130k_power_law_concept_gaussian_vectors.npz' 
    #image_feature_file = '../data/mscoco/mscoco_subset_130k_res34_embed512dim.npz'
    # XXX
    exp_dir = 'exp/feb_5_mscoco20k_synthetic_concept_%d_mixture=%d/' % (model_configs['Kmax'], model_configs['Mmax'])
    print(exp_dir)      
    model = ImagePhoneMixtureWordDiscoverer(speech_feature_file, image_feature_file, model_configs=model_configs, model_name=exp_dir+'image_phone_mixture')
    begin_time = time.time()
    model.train_using_EM(num_iterations=10)
    print('Take %.5f s to finish training !' % (time.time() - begin_time))
    model.print_alignment(exp_dir + 'image_phone')
    print('Take %.5f s to finish decoding !' % (time.time() - begin_time))

