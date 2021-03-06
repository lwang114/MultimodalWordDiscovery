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
class ImageAudioGMMWordDiscoverer:
  def __init__(self, speech_feature_file, image_feature_file, model_configs, model_name='image_audio_word_discoverer'):
    self.model_name = model_name
    self.a_corpus = []
    self.v_corpus = []
    
    # Prior parameters
    self.Kmax = model_configs.get('Kmax', 1000) # Maximum number of vocabs
    self.Mmax = model_configs.get('Mmax', 5) # Maximum number of mixtures per word
    
    self.embed_dim = model_configs.get('embedding_dim', 130)
    
    # XXX: Assume hyperparameters are in log scale 
    g_sb00 = model_configs.get('gamma_sb0', 1.) # Stick-breaking beta distribution parameter
    self.g_sb0 = -np.inf * np.ones((self.Kmax, 2))
    for k in range(self.Kmax):
      self.g_sb0[k, 0] = np.log(g_sb00)
      if k < self.Kmax - 1:
        self.g_sb0[k, 1] = np.log(g_sb00 * (self.Kmax - k - 1))
         
    self.g_a0 = model_configs.get('gamma_a', 0.1)
    self.g_a0 = np.log(self.g_a0)
    self.g_ma0 = model_configs.get('gamma_ma', 0.1) # Audio mixture weight prior parameter
    self.g_ma0 = np.log(self.g_ma0)
    self.g_mv0 = model_configs.get('gamma_mv', 0.1) # Visual mixture weight prior parameter
    self.g_mv0 = np.log(self.g_mv0)
    self.k_a0 = model_configs.get('k_a', 0.1) # Ratio between obs variance and prior variance
    self.k_a0 = np.log(self.k_a0)
    self.k_v0 = model_configs.get('k_v', 0.1) # Ratio between obs variance and prior variance
    self.k_v0 = np.log(self.k_v0)
    self.b_a0 = model_configs.get('beta_a0', 0.1) # Parameters for the gamma prior of the inverse variances for audio
    self.b_a1 = model_configs.get('beta_a1', 0.1) # Parameters for the gamma prior of the inverse variances for audio 
    self.b_v0 = model_configs.get('beta_v0', 0.1) # Parameters for the gamma prior of the inverse variances for image
    self.b_v1 = model_configs.get('beta_v1', 0.1) # Parameters for the gamma prior of the inverse variances for image

    alignments = model_configs.get('alignments', None)
    segmentations = model_configs.get('segmentations', None)
    self.has_null = model_configs.get('has_null', True)
    self.init_method = model_configs.get('initialize_method', 'rand')
    self.fixed_variance = model_configs.get('fixed_variance', False)

    self.read_corpus(speech_feature_file, image_feature_file, segmentations)

    self.len_probs = {}
    self.concept_prior = None 
    self.align_init = {}
    self.audio_obs_model = {'weights': None,
                            'means': None,
                            'variance': None,
                            'n_mixtures': None
                            }
    self.image_obs_model = {'weights': None,
                            'means': None,
                            'n_mixures': None
                            }    
    
  def read_corpus(self, speech_feat_file, image_feat_file, segmentations=None):
    a_npz = np.load(speech_feat_file)
    v_npz = np.load(image_feat_file)
    
    if segmentations is not None: 
      # XXX
      a_feat_corpus = [np.array(a_npz[k]) for k in sorted(a_npz, key=lambda x:int(x.split('_')[-1]))]
      self.a_corpus = [self.get_sent_embeds(a_feat, segmentation) for a_feat, segmentation in zip(a_feat_corpus, segmentations)]
    else:
      # XXX
      self.a_corpus = [np.array(a_npz[k]) for k in sorted(a_npz, key=lambda x:int(x.split('_')[-1]))] 

    self.v_corpus = [np.array(v_npz[k]) for k in sorted(v_npz, key=lambda x:int(x.split('_')[-1]))] 
    #print('a_corpus: ', self.a_corpus)
    #print('v_corpus: ', self.v_corpus)
    if len(self.a_corpus[0].shape) == 1:
      self.a_corpus = [afeat[np.newaxis, :] for afeat in self.a_corpus]
      self.v_corpus = [vfeat[np.newaxis, :] for vfeat in self.v_corpus]

    self.image_feat_dim = self.v_corpus[0].shape[-1]  
    for ex, (afeat, vfeat) in enumerate(zip(self.a_corpus, self.v_corpus)):
      if afeat.shape[-1] == 0:
        print('ex: ', ex)
        print('afeat empty: ', afeat.shape) 
        self.a_corpus[ex] = np.zeros((1, self.embed_dim))

      if vfeat.shape[-1] == 0:
        print('ex: ', ex)
        print('vfeat empty: ', vfeat.shape) 
        self.v_corpus[ex] = np.zeros((1, self.image_feat_dim))
      
    if self.has_null:
      self.v_corpus = [np.concatenate((np.zeros((1, self.image_feat_dim)), vfeat), axis=0) for vfeat in self.v_corpus]  
    
    assert len(self.v_corpus) == len(self.a_corpus)

  def estimate_prior_mean_var(self):
    n_frames_a, n_frames_v = 0., 0.
    self.mu_a0, self.mu_v0 = np.zeros((self.embed_dim,)), np.zeros((self.image_feat_dim,))
    for afeat, vfeat in zip(self.a_corpus, self.v_corpus):
      #print('afeat.shape, vfeat.shape: ', afeat.shape, vfeat.shape)
      self.mu_a0 += np.sum(afeat, axis=0)
      self.mu_v0 += np.sum(vfeat, axis=0)
      n_frames_a += afeat.shape[0]
      n_frames_v += vfeat.shape[0]

    self.mu_a0 /= n_frames_a
    self.mu_v0 /= n_frames_v

    c = 3.  
    if self.fixed_variance:
      self.fixed_variance_a = 0. # XXX: Assume fixed variances  
      self.fixed_variance_v = 0. # XXX: Assume fixed variances  
      for afeat, vfeat in zip(self.a_corpus, self.v_corpus):
        self.fixed_variance_a += np.sum((afeat - self.mu_a0)**2, axis=0) / (c * n_frames_a * self.embed_dim)
        self.fixed_variance_v += np.sum((vfeat - self.mu_v0)**2, axis=0) / (c * n_frames_v * self.image_feat_dim)
      self.audio_obs_model['variances'] = np.tile(self.fixed_variance_a[np.newaxis, np.newaxis, :], (self.Kmax, self.Mmax))
      self.image_obs_model['variances'] = np.tile(self.fixed_variance_v[np.newaxis, np.newaxis, :], (self.Kmax, self.Mmax))
    else:
      self.audio_obs_model['variances'] = self.b_a1 * np.ones((self.Kmax, self.Mmax, self.embed_dim))
      self.image_obs_model['variances'] = self.b_v1 * np.ones((self.Kmax, self.Mmax, self.image_feat_dim))
      for afeat, vfeat in zip(self.a_corpus, self.v_corpus):
        self.audio_obs_model['variances'] += np.sum((afeat - self.mu_a0)**2, axis=0) / (c * n_frames_a * self.embed_dim)
        self.image_obs_model['variances'] += np.sum((vfeat - self.mu_v0)**2, axis=0) / (c * n_frames_v * self.image_feat_dim)

      self.b_a1 = self.b_a0 * np.mean(self.audio_obs_model['variances'])
      self.b_v1 = self.b_v0 * np.mean(self.image_obs_model['variances'])

    # XXX: Check if the means and variance look reasonable
    #print('self.mu_a0.shape, self.mu_v0.shape: ', self.mu_a0.shape, self.mu_v0.shape)
    #print('Var(a), Var(v): ', self.fixed_variance_a, self.fixed_variance_v) 
  
  def initialize_model(self, alignments=None):
    begin_time = time.time()
    self.compute_translation_length_probabilities()

    for lv in self.len_probs:
      self.align_init[lv] = np.log(1./lv) * np.ones((lv,))

    self.vs = np.zeros((self.Kmax,)) 
    for k in range(self.Kmax):
      self.vs[k] = np.exp(self.g_sb0[k, 0]) / (np.exp(self.g_sb0[k, 0]) + np.exp(self.g_sb0[k, 1]))
    self.concept_prior = np.log(compute_stick_break_prior(self.vs)) 
    #print('concept_prior', np.exp(self.concept_prior))
    
    self.audio_obs_model['weights'] = np.log(1./self.Mmax) * np.ones((self.Kmax, self.Mmax)) 
    self.audio_obs_model['n_mixtures'] = self.Mmax * np.ones((self.Kmax,))
    self.image_obs_model['weights'] = np.log(1./self.Mmax) * np.ones((self.Kmax, self.Mmax))
    self.image_obs_model['n_mixtures'] = self.Mmax * np.ones((self.Kmax,))
    self.audio_obs_model['means'], self.image_obs_model['means'] = self.initialize_mixture_means(method=self.init_method)
    self.estimate_prior_mean_var()
    
    # XXX
    #self.audio_obs_model['means'] = deepcopy(self.image_obs_model['means'])
   
    # Initialize hyperparameters for the approximate parameter posteriors
    self.g_aN = {l: (self.g_a0 - np.log(l)) * np.ones((l,)) for l in self.len_probs}  # size-L dict of Nv-d array
    self.g_sbN = deepcopy(self.g_sb0) # K-d array
    self.g_maN = self.g_ma0 * np.ones((self.Kmax, self.Mmax)) # Ma-d array
    self.g_mvN = self.g_mv0 * np.ones((self.Kmax, self.Mmax)) # Mv-d array 
    self.k_aN = self.k_a0 * np.ones((self.Kmax, self.Mmax)) 
    self.k_vN = self.k_v0 * np.ones((self.Kmax, self.Mmax)) 

    # Initialize the approximate posteriors
    self.counts_concept = None #[np.array([deepcopy(self.concept_prior) for _ in range(vfeat.shape[0])]) for vfeat in self.v_corpus]
    self.counts_audio_mixture = None 
    self.counts_image_mixture = None 
    if alignments is None:
      self.counts_align = [np.log(1./len(vfeat)) * np.ones((len(afeat), len(vfeat))) for afeat, vfeat in zip(self.a_corpus, self.v_corpus)] 
    else:
      self.counts_align = []
      for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)): 
        a = np.asarray(a, dtype=int)
        T = a.shape[0]
        n_state = a.shape[1]
        count_align = -np.inf*np.ones((T, n_state)) 
        count_align[a.tolist()] = 0.
        self.counts_align.append(count_align)

    # Initialize the expressions involving the digamma function
    self.update_digamma_functions()
    print('takes %.5f s to finish initialization' % (time.time() - begin_time))

  def initialize_mixture_means(self, method='rand'):
    if method == 'rand': 
      #a_corpus_concat = np.concatenate(self.a_corpus, axis=0)
      #n_frames_a = a_corpus_concat.shape[0]
      #v_corpus_concat = np.concatenate(self.v_corpus, axis=0)
      #n_frames_v = v_corpus_concat.shape[0]
      n_corpus = len(self.a_corpus)
    
      cluster_centers_a = -np.inf * np.ones((self.Kmax, self.Mmax, self.embed_dim))
      cluster_centers_v = -np.inf * np.ones((self.Kmax, self.Mmax, self.image_feat_dim))
      
      for k in range(self.Kmax):
        for m in range(self.Mmax):
          #i_a = np.random.randint(0, n_frames_a-1)
          #i_v = np.random.randint(0, n_frames_v-1)
          #cluster_centers_a[k, m] = a_corpus_concat[i_a]
          #cluster_centers_v[k, m] = v_corpus_concat[i_v] 
          i_ex = np.random.randint(0, n_corpus-1)
          if len(self.a_corpus[i_ex]) <= 1:
            i_a = 0
            i_v = 0
          else:
            i_a = np.random.randint(0, len(self.a_corpus[i_ex])-1)
            i_v = np.random.randint(0, len(self.v_corpus[i_ex])-1)
          cluster_centers_a[k, m] = self.a_corpus[i_ex][i_a]
          cluster_centers_v[k, m] = self.v_corpus[i_ex][i_v]

      return cluster_centers_a, cluster_centers_v
    else:
      kmeans_a = KMeans(n_clusters=self.Mmax * self.Kmax, max_iter=10).fit(np.concatenate(self.a_corpus, axis=0))
      kmeans_v = KMeans(n_clusters=self.Mmax * self.Kmax, max_iter=10).fit(np.concatenate(self.v_corpus, axis=0))
      #print('kmeans_a.means.shape, kmeans_v.shape: ', kmeans_a.cluster_centers_.shape, kmeans_v.cluster_centers_.shape)
      return kmeans_a.cluster_centers_.reshape(self.Kmax, self.Mmax, -1), kmeans_v.cluster_centers_.reshape(self.Kmax, self.Mmax, -1)

  def update_concept_counts(self, counts_align_init, probs_afeat_given_zm, probs_vfeat_given_zm):
    T = len(probs_afeat_given_zm)
    n_state = len(probs_vfeat_given_zm)
    counts_concept = -np.inf * np.ones((n_state, self.Kmax))
    
    probs_afeat_given_z = []
    for i in range(n_state):
      counts_concept[i] = deepcopy(self.digammas_concept)
    
    probs_afeat_given_z = np.sum(self.digammas_ma + probs_afeat_given_zm, axis=-1) 
    counts_concept += np.exp(counts_align_init).T @ probs_afeat_given_z 
    counts_concept += np.sum(self.digammas_mv + probs_vfeat_given_zm, axis=-1)
 
    # Normalize
    for i in range(n_state):
      counts_concept[i] -= logsumexp(counts_concept[i])
    
    return counts_concept

  def update_audio_mixture_counts(self, probs_afeat_given_zm):
    counts_audio_mixture = self.digammas_ma + probs_afeat_given_zm
    #print('after adding the digamma_ma: ', counts_audio_mixture)
    # Normalize
    norm_factors = logsumexp(counts_audio_mixture, axis=-1) 
    #print('norm_factors for audio mixture:  ', norm_factors)
    counts_audio_mixture = np.transpose(np.transpose(counts_audio_mixture, (2, 0, 1)) - norm_factors, (1, 2, 0)) 
    return counts_audio_mixture 

  def update_image_mixture_counts(self, probs_vfeat_given_zm):
    counts_image_mixture = self.digammas_mv + probs_vfeat_given_zm
    # Normalize
    norm_factors = logsumexp(counts_image_mixture, axis=-1) 
    counts_image_mixture = np.transpose(np.transpose(counts_image_mixture, (2, 0, 1)) - norm_factors, (1, 2, 0)) 
    return counts_image_mixture 

  def update_alignment_counts(self, log_probs_a_given_zm, counts_concept, counts_audio_mixture):
    n_states = counts_concept.shape[0]
    counts_align = self.log_probs_afeat_given_i(log_probs_a_given_zm, counts_concept, counts_audio_mixture)
    counts_align += self.digammas_align[n_states]
    counts_align = (counts_align.T - logsumexp(counts_align, axis=-1)).T

    assert not np.isnan(counts_align).any()
    return counts_align 
        
  def update_posterior_hyperparameters(self):
    new_g_maN = (self.g_ma0 - np.log(self.Mmax)) * np.ones((self.Kmax, self.Mmax))
    new_g_mvN = (self.g_mv0 - np.log(self.Mmax)) * np.ones((self.Kmax, self.Mmax))
    new_g_aN = {l: (self.g_a0 - np.log(l)) * np.ones((l,)) for l in self.len_probs}
    new_g_sbN = deepcopy(self.g_sb0) 
    new_k_aN = self.k_a0 * np.ones((self.Kmax, self.Mmax))
    new_k_vN = self.k_v0 * np.ones((self.Kmax, self.Mmax))
    
    aggregated_counts = [logsumexp(counts, axis=0) for counts in self.counts_audio_mixture]
    new_g_maN = logsumexp(np.array([new_g_maN]+aggregated_counts), axis=0)
    new_k_aN = logsumexp(np.array([new_k_aN]+aggregated_counts), axis=0)

    aggregated_counts = None
    aggregated_counts = [logsumexp(counts, axis=0) for counts in self.counts_image_mixture]
    new_g_mvN = logsumexp(np.array([new_g_mvN]+aggregated_counts), axis=0) 
    new_k_vN = logsumexp(np.array([new_k_vN]+aggregated_counts), axis=0)
   
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

    self.g_maN = new_g_maN
    self.g_mvN = new_g_mvN
    self.g_aN = new_g_aN
    self.g_sbN = new_g_sbN
    self.k_aN = new_k_aN 
    self.k_vN = new_k_vN

  def update_digamma_functions(self):
    self.digammas_align = {l:None for l in self.g_aN} 
    
    for l in self.g_aN:
      self.digammas_align[l] = digamma(np.exp(self.g_aN[l])) - digamma(np.exp(logsumexp(self.g_aN[l])))
      #print('self.digammas_align: ', self.digammas_align[l])

    self.digammas_concept = np.zeros((self.Kmax,))
    self.digammas_ma = np.zeros((self.Kmax, self.Mmax))
    self.digammas_mv = np.zeros((self.Kmax, self.Mmax))
    
    for k in range(self.Kmax):
      for j in range(k-1):
        self.digammas_concept[k] += digamma(np.exp(self.g_sbN[j, 1])) - digamma(np.exp(logsumexp(self.g_sbN[j])))
      self.digammas_concept[k] += digamma(np.exp(self.g_sbN[k, 0])) - digamma(np.exp(logsumexp(self.g_sbN[k])))
     
      for m in range(self.Mmax):
        self.digammas_ma[k] = digamma(np.exp(self.g_maN[k])) - digamma(np.exp(logsumexp(self.g_maN[k])))
      
      for m in range(self.Mmax):
        self.digammas_mv[k] = digamma(np.exp(self.g_mvN[k])) - digamma(np.exp(logsumexp(self.g_mvN[k])))
    #print('self.digammas_concept: ', np.exp(self.digammas_concept))
    #print('self.digammas_ma: ', self.digammas_ma)
    #print('self.digammas_mv: ', self.digammas_mv)
         
  def update_concept_alignment_model(self, ):
    vs = np.exp(self.g_sbN[:, 0]) / (np.exp(self.g_sbN[:, 0]) + np.exp(self.g_sbN[:, 1]))
    
    self.concept_prior = None
    self.concept_prior = np.log(compute_stick_break_prior(vs))
    
    for l in self.g_aN:
      self.align_init[l] = self.g_aN[l] - logsumexp(self.g_aN[l])

  # TODO: Make the counts concept a global variable later 
  def update_observation_models(self):
    self.audio_obs_model['weights'] = -np.inf * np.ones(self.g_maN.shape)
    self.image_obs_model['weights'] = -np.inf * np.ones(self.g_mvN.shape)
    exp_audio_mixture_means = np.tile(np.exp(self.k_a0) * self.mu_a0, (self.audio_obs_model['means'].shape[0], self.audio_obs_model['means'].shape[1], 1))
    exp_image_mixture_means = np.tile(np.exp(self.k_v0) * self.mu_v0, (self.image_obs_model['means'].shape[0], self.image_obs_model['means'].shape[1], 1))
    # XXX: factor of 2 for consistency of definitions
    exp_audio_mixture_variances = 2*self.b_a1 * np.ones((self.Kmax, self.Mmax, self.embed_dim))
    exp_image_mixture_variances = 2*self.b_v1 * np.ones((self.Kmax, self.Mmax, self.image_feat_dim))
    exp_num_audio_mixture = np.zeros((self.Kmax, self.Mmax))
    exp_num_image_mixture = np.zeros((self.Kmax, self.Mmax))

    for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
      for k in range(self.Kmax):
        counts_z_a = logsumexp(self.counts_align[ex] + self.counts_concept[ex][:, k], axis=-1)
        counts_ma_z = counts_z_a + self.counts_audio_mixture[ex][:, k, :].T
        counts_mv_z = self.counts_concept[ex][:, k] + self.counts_image_mixture[ex][:, k, :].T
        #print('self.counts_align[ex]: ', self.counts_align[ex])
        #print('counts_z_a, counts_ma_z, counts_mv_z: ', np.exp(counts_z_a), np.exp(counts_ma_z), np.exp(counts_mv_z))
        exp_audio_mixture_means[k] += np.exp(counts_ma_z) @ afeats
        exp_image_mixture_means[k] += np.exp(counts_mv_z) @ vfeats

        exp_num_audio_mixture[k] += np.exp(logsumexp(counts_ma_z, axis=1))
        exp_num_image_mixture[k] += np.exp(logsumexp(counts_mv_z, axis=1))
    
    for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
      for k in range(self.Kmax):
        counts_z_a = logsumexp(self.counts_align[ex] + self.counts_concept[ex][:, k], axis=-1)
        counts_ma_z = (counts_z_a + self.counts_audio_mixture[ex][:, k, :].T)
        counts_mv_z = (self.counts_concept[ex][:, k] + self.counts_image_mixture[ex][:, k, :].T)
        for m in range(self.Mmax):
          exp_audio_mixture_variances[k, m] += np.exp(counts_ma_z[m]) @ ((afeats - self.audio_obs_model['means'][k, m]) ** 2) / (2.*self.b_a0 + exp_num_audio_mixture[k, m])
          exp_image_mixture_variances[k, m] += np.exp(counts_mv_z[m]) @ ((vfeats - self.image_obs_model['means'][k, m]) ** 2) / (2.*self.b_v0 + exp_num_image_mixture[k, m]) 

    #print('exp_audio_mixture_means.shape, exp_num_audio_mixture.shape: ', exp_audio_mixture_means.shape, exp_num_audio_mixture.shape)
    if np.min(exp_audio_mixture_variances) <= 0:
      print('audio mixture variance is 0: ', np.argmin(exp_audio_mixture_variances))
    if np.min(exp_image_mixture_variances) <= 0:
      print('image mixture variance is 0: ', np.argmin(exp_image_mixture_variances))

    self.audio_obs_model['variances'] = exp_audio_mixture_variances 
    self.image_obs_model['variances'] = exp_image_mixture_variances 
    #print('exp_audio_mixture_means.shape, exp_num_audio_mixture.shape: ', exp_audio_mixture_means.shape, exp_num_audio_mixture.shape)

    self.audio_obs_model['means'] = np.transpose(np.transpose(exp_audio_mixture_means, (2, 0, 1)) / (np.exp(self.k_a0) + exp_num_audio_mixture), (1, 2, 0))
    self.image_obs_model['means'] = np.transpose(np.transpose(exp_image_mixture_means, (2, 0, 1)) / (np.exp(self.k_v0) + exp_num_image_mixture), (1, 2, 0))
    # Normalize
    self.audio_obs_model['weights'] = (self.g_maN.T - logsumexp(self.g_maN, axis=1)).T
    self.image_obs_model['weights'] = (self.g_mvN.T - logsumexp(self.g_mvN, axis=1)).T 

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
      new_counts_audio_mixture = []
      new_counts_image_mixture = []
      new_counts_align = [] 

      for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
        log_probs_a_given_zm = self.log_probs_afeat_given_z_m(afeats)
        log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)
        #print('afeats: ', afeats)
        #print('vfeats: ', vfeats)
        #print('log_probs_a_given_zm: ', log_probs_a_given_zm)
        #print('log_probs_v_given_zm: ', log_probs_v_given_zm)
        #print('counts_audio_mixture: ', counts_audio_mixture)
        #print('counts_image_mixture: ', counts_image_mixture)
        #print('sum(counts_audio_mixture): ', np.sum(np.exp(counts_audio_mixture), axis=-1))
        #print('sum(counts_image_mixture): ', np.sum(np.exp(counts_image_mixture), axis=-1))
        
        # Estimate counts
        counts_concept = self.update_concept_counts(self.counts_align[ex], log_probs_a_given_zm, log_probs_v_given_zm)
        #print('counts_concept: ', np.exp(counts_concept))
        new_counts_concept.append(counts_concept)
       
        counts_audio_mixture = self.update_audio_mixture_counts(log_probs_a_given_zm)
        counts_image_mixture = self.update_image_mixture_counts(log_probs_v_given_zm)
        new_counts_audio_mixture.append(counts_audio_mixture)
        new_counts_image_mixture.append(counts_image_mixture)

        counts_align = self.update_alignment_counts(log_probs_a_given_zm, counts_concept, counts_audio_mixture) 
        new_counts_align.append(counts_align)
             
        #print('counts_align: ', counts_align)
        #print('sum(counts_align_init): ', np.sum(np.exp(counts_align), axis=-1))    
      
      print('Take %.5f s for E step' % (time.time() - begin_time)) 
      begin_time = time.time()
      self.counts_concept = deepcopy(new_counts_concept)
      self.counts_audio_mixture = deepcopy(new_counts_audio_mixture)
      self.counts_image_mixture = deepcopy(new_counts_image_mixture)
      self.counts_align = deepcopy(new_counts_align)
      # Update posterior hyperparameters
      self.update_posterior_hyperparameters()
      self.update_digamma_functions()
      #print('new_g_maN: ', self.g_maN)
      #print('new_g_mvN: ', self.g_mvN)
      #print('new_g_aN: ', [self.g_aN[l] for l in new_g_aN])
      #print('new_g_sbN: ', np.exp(self.g_sbN))

      # Update parameters
      self.update_concept_alignment_model()
      self.update_observation_models()
      #print('audio_obs_means: ', self.audio_obs_model['means'])
      #print('image_obs_means: ', self.image_obs_model['means'])
      #print('self.mu_a0, self.mu_v0: ', self.mu_a0, self.mu_v0)
      #print('sum(audio_obs_means): ', np.sum(self.audio_obs_model['means'], axis=0))
      #print('sum(image_obs_means): ', np.sum(self.image_obs_model['means'], axis=0))      
      #print('sum(p(m_at|z_t)): ', np.sum(np.exp(self.audio_obs_model['weights']), axis=-1))
      #print('sum(p(m_vt|z_t)): ', np.sum(np.exp(self.image_obs_model['weights']), axis=-1))

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
      log_probs_a_given_zm = self.log_probs_afeat_given_z_m(afeats)
      log_probs_v_given_z = logsumexp(log_probs_v_given_zm + self.image_obs_model['weights'], axis=-1)
      log_probs_v = logsumexp(self.concept_prior + log_probs_v_given_z, axis=1)
      # TODO: Check sum = 1  
      log_probs_z_given_v = ((self.concept_prior + log_probs_v_given_z).T - log_probs_v).T
      #print('sum(p(z)): ', np.sum(np.exp(self.concept_prior)))
      #print('sum(p(z|v)): ', np.sum(np.exp(log_probs_z_given_v), axis=1))
      log_prob_v_tot = np.sum(log_probs_v)
      log_probs_a_i_given_v = self.log_prob_afeats_align_given_vfeats(log_probs_a_given_zm, log_probs_z_given_v)
      #print(log_probs_a_i_given_v.shape)
      log_prob_a_given_v_tot = logsumexp(log_probs_a_i_given_v.flatten())
      ll += 1. / len(self.a_corpus) * (log_prob_v_tot + log_prob_a_given_v_tot)
    return ll
  
  def compute_ELBO(self):
    elbo = 0.
    vs = 1./2 * np.ones((self.Kmax,)) 
    vs[-1] = 1.
    init_concept_prior = np.log(compute_stick_break_prior(vs)) 

    kl_div_ma_prob = KL_divergence(self.audio_obs_model['weights'], np.log(1. / self.Mmax) * np.ones((self.Kmax, self.Mmax)), log_prob=True)
    kl_div_mv_prob = KL_divergence(self.image_obs_model['weights'], np.log(1. / self.Mmax) * np.ones((self.Kmax, self.Mmax)), log_prob=True)
    #print('kl_div_ma_prob, kl_div_mv_prob: ', kl_div_ma_prob, kl_div_mv_prob)
    
    for n_state in self.len_probs:
      kl_div_align_prob = KL_divergence(self.align_init[n_state], np.log(1. / n_state) * np.ones((n_state,)), log_prob=True)
      elbo -= kl_div_align_prob
    
    kl_div_mean_a = gaussian_KL_divergence(self.audio_obs_model['means'], 
                                            self.audio_obs_model['variances'],
                                            np.tile(self.mu_a0[np.newaxis, np.newaxis, :], (self.Kmax, self.Mmax, 1)),
                                            self.b_a1 / self.b_a0 * np.ones((self.Kmax, self.Mmax, self.embed_dim)),
                                            cov_type='diag')
    kl_div_mean_v = gaussian_KL_divergence(self.image_obs_model['means'], 
                                            self.image_obs_model['variances'],
                                            np.tile(self.mu_v0[np.newaxis, np.newaxis, :], (self.Kmax, self.Mmax, 1)), 
                                            self.b_v1 / self.b_v0 * np.ones((self.Kmax, self.Mmax, self.image_feat_dim)), 
                                            cov_type='diag')
    kl_div_concept_prob = KL_divergence(self.concept_prior, init_concept_prior)  
    kl_div_parameters = kl_div_mean_a + kl_div_mean_v + kl_div_ma_prob + kl_div_mv_prob + kl_div_concept_prob
    elbo -= kl_div_parameters  
    #print('KL divergence of audio means: ', kl_div_mean_a)
    #print('KL divergence of image means: ', kl_div_mean_v)

    for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
      T = afeats.shape[0]
      n_state = vfeats.shape[0]
      log_probs_a_given_zm = self.log_probs_afeat_given_z_m(afeats)
      log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)
      log_probs_afeat_given_i = self.log_probs_afeat_given_i(log_probs_a_given_zm, self.counts_concept[ex], self.counts_audio_mixture[ex])
      log_probs_afeat = np.sum(np.exp(self.counts_align[ex]) * log_probs_afeat_given_i)  
      log_probs_vfeat = np.sum(np.tile(np.exp(self.counts_concept[ex])[:, :, np.newaxis], (1, 1, self.Mmax)) * np.exp(self.counts_image_mixture[ex]) * log_probs_v_given_zm) 
      elbo += log_probs_afeat + log_probs_vfeat
      
      kl_div_ma = KL_divergence(self.counts_audio_mixture[ex], np.tile(self.audio_obs_model['weights'][np.newaxis, :, :], (T, 1, 1)), log_prob=True)
      kl_div_mv = KL_divergence(self.counts_image_mixture[ex], np.tile(self.image_obs_model['weights'][np.newaxis, :, :], (n_state, 1, 1)), log_prob=True)
      
      kl_div_align = KL_divergence(self.counts_align[ex], self.align_init[n_state][np.newaxis, :])
      kl_div_concept = KL_divergence(self.counts_concept[ex], np.tile(self.concept_prior[np.newaxis, :], (n_state, 1)))
 
      # XXX
      #kl_div_ma = 0.
      #kl_div_mv = 0.
      kl_div_latent_variables = kl_div_ma + kl_div_mv + kl_div_align + kl_div_concept  
      elbo -= kl_div_latent_variables 

    return elbo

  def log_prob_afeats_align_given_vfeats(self, log_probs_a_given_zm, log_probs_z_given_v):
    T = len(log_probs_a_given_zm)
    n_states = len(log_probs_z_given_v)
    log_probs_a_i_given_v = -np.inf * np.ones((T, n_states)) 
    log_probs_a_given_z = logsumexp(self.audio_obs_model['weights'] + log_probs_a_given_zm, axis=-1)
    for i in range(n_states):
      log_probs_a_i_given_v[:, i] = self.align_init[n_states][i] + logsumexp(log_probs_z_given_v[i] + log_probs_a_given_z, axis=-1)
    
    return log_probs_a_i_given_v
    
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
     
  def log_prob_afeat_given_z_m(self, afeat, k, m):
    return compute_diagonal_gaussian(afeat, self.audio_obs_model['means'][k, m], self.audio_obs_model['variances'][k, m], log_prob=True)

  def log_prob_vfeat_given_z_m(self, vfeat, k, m):
    return compute_diagonal_gaussian(vfeat, self.image_obs_model['means'][k, m], self.image_obs_model['variances'][k, m], log_prob=True)

  #def log_prob_afeat_given_z_m(self, afeat):
  # Compute approximate p(afeat_t|i) for all t and i and store them in a T x N(v) matrix 
  def log_probs_afeat_given_i(self, log_probs_a_given_zm, counts_concept, counts_audio_mixture):
    return np.sum(np.exp(counts_audio_mixture) * log_probs_a_given_zm, axis=-1) @ np.exp(counts_concept).T   
  
  # Compute approximate p(afeat_t|z_i(t), m_t) for all t, z_i(t) and m and store them in a T x K x M matrix
  def log_probs_afeat_given_z_m(self, afeats):
    log_probs = []
    for k in range(self.Kmax):
      log_probs_m = []
      for m in range(self.Mmax):
        log_probs_m.append(self.log_prob_afeat_given_z_m(afeats, k, m))
      log_probs.append(log_probs_m)
    return np.transpose(np.asarray(log_probs), (2, 0, 1))
  
  def log_probs_vfeat_given_z_m(self, vfeats):
    n_states = vfeats.shape[0]
    log_probs = []
    
    for m in range(self.Mmax):   
      log_probs_m = []
      for i in range(n_states):
        log_probs_m.append(compute_diagonal_gaussian(vfeats[i], self.image_obs_model['means'][:, m], self.image_obs_model['variances'][:, m, :], log_prob=True))
      log_probs.append(log_probs_m)

    return np.transpose(np.asarray(log_probs), (1, 2, 0))
  
  # Compute approximate p(afeat_t, vfeat_i|i) for all t and i and store them in a T x N(v) matrix 
  def log_probs_afeat_vfeat_given_i(self, log_probs_a_given_zm, log_probs_v_given_zm, counts_concept, counts_audio_mixture, counts_image_mixture):
    log_probs = []
    n_states = log_probs_v_given_zm.shape[0]
    log_probs_a_given_i = self.log_probs_afeat_given_i(log_probs_a_given_zm, counts_concept, counts_audio_mixture)
    
    log_probs_v = -np.inf*np.ones((n_states,))
    for i in range(n_states):
      log_probs_v[i] = np.sum(np.exp(counts_concept[np.newaxis, i]) @ (np.exp(counts_image_mixture[i]) * log_probs_v_given_zm[i]), axis=1)
    #print('log_probs_v: ', log_probs_v)
    
    return log_probs_a_given_i + log_probs_v
 
  def get_sent_embeds(self, x, segmentation, frame_dim=12):
    if segmentations is None:
      return x
    n_words = len(segmentation) - 1
    embeddings = []
    for i_w in range(n_words):
      seg = x[segmentation[i_w]:segmentation[i_w+1]]
      #print("seg.shape", seg.shape)
      #print("seg:", segmentation[i_w+1])
      #print("embed of seg:", self.embed(seg))
      embeddings.append(self.embed(seg, frame_dim=frame_dim))  
    return np.array(embeddings)
   
  def embed(self, y, frame_dim=None, technique="resample"):
    #assert self.embed_dim % self.audio_feat_dim == 0
    if frame_dim:
      y = y[:, :frame_dim].T
    else:
      y = y.T
      frame_dim = self.audio_feat_dim

    n = int(self.embed_dim / frame_dim)
    if y.shape[0] == 1: 
      y_new = np.repeat(y, n)   

    #if y.shape[0] <= n:
    #  technique = "interpolate" 
         
    #print(xLen, self.embed_dim / self.feat_dim)
    if technique == "interpolate":
        x = np.arange(y.shape[1])
        f = interpolate.interp1d(x, y, kind="linear")
        x_new = np.linspace(0, y.shape[1] - 1, n)
        y_new = f(x_new).flatten(ORDER) #.flatten("F")
    elif technique == "resample":
        y_new = signal.resample(y.astype("float32"), n, axis=1).flatten(ORDER) #.flatten("F")
    elif technique == "rasanen":
        # Taken from Rasenen et al., Interspeech, 2015
        n_frames_in_multiple = int(np.floor(y.shape[1] / n)) * n
        y_new = np.mean(
            y[:, :n_frames_in_multiple].reshape((d_frame, n, -1)), axis=-1
            ).flatten(ORDER) #.flatten("F")
    return y_new
  
  # Align the i-th example in the training set
  def align_i(self, i): 
    T = self.a_corpus[i].shape[0]
    n_states = self.v_corpus[i].shape[0]
     
    concept_scores = self.counts_concept[i]
    align_scores = self.counts_align[i]
    best_alignment = np.argmax(align_scores, axis=1).tolist()
    best_concepts = np.argmax(concept_scores, axis=1).tolist()
    
    return best_concepts, concept_scores.tolist(), best_alignment, align_scores.tolist()

  # TODO: Convert to be without Markov chain 
  # TODO: Check if the parameters and the hyperparameters of the alignment transition still remains 
  def align(self, afeats, vfeats):
    if self.has_null:
      vfeats = np.concatenate((np.zeros((1, self.image_feat_dim)), vfeats), axis=0)

    T = len(afeats)
    n_states = len(vfeats)   
    back_ptrs = np.zeros((T, n_states), dtype=int)
    
    align_scores = -np.inf * np.ones((T, n_states))
    concept_scores = None 
    align_probs = []

    log_probs_a_given_zm = self.log_probs_afeat_given_z_m(afeats)
    log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)
    log_probs_v_given_z = logsumexp(log_probs_v_given_zm + self.audio_obs_model['weights'], axis=-1)  
    log_probs_v = logsumexp(self.concept_prior + log_probs_v_given_z, axis=1)  
    log_probs_z_given_v = ((self.concept_prior + log_probs_v_given_z).T - log_probs_v).T    
    log_probs_a_given_z = logsumexp(self.audio_obs_model['weights'] + log_probs_a_given_zm, axis=-1)
      
    for i in range(n_states):
      align_scores[:, i] = self.align_init[n_states][i] + logsumexp(log_probs_z_given_v[i] + log_probs_a_given_z, axis=-1)

    best_alignment = np.argmax(align_scores, axis=1).tolist()

    concept_scores = -np.inf * np.ones((n_states, self.Kmax)) 
    align_matrix = np.zeros((T, n_states))
    align_matrix[best_alignment] = 1.

    for k in range(n_states):
      concept_scores[k] = deepcopy(log_probs_z_given_v[k])
      log_prob_a_all_given_z = np.sum(align_matrix[:, k] * log_probs_a_given_z.T, axis=1) 
      concept_scores[k] += log_prob_a_all_given_z

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
  assert np.min(np.diagonal(cov1, axis1=-1, axis2=-2)) > 0. and np.min(np.diagonal(cov2, axis1=-1, axis2=-2)) > 0.
  #print('mean and cov shapes: ', mean1.shape, cov1.shape, mean2.shape, cov2.shape)
  assert mean1.shape[-1] == cov1.shape[-1] and mean2.shape[-1] == cov2.shape[-1] and mean1.shape == mean2.shape

  if cov_type == 'diag':
    d = mean1.shape[-1]
    tr_cov2_inv_cov1 = np.sum(cov1 / cov2, axis=-1)
    mahalanobis = np.sum((mean2-mean1)**2 / cov2, axis=-1)
    log_det_cov1_inv_cov2 = np.sum(np.log(cov2 / cov1), axis=-1)
    return np.sum(1./2 * (tr_cov2_inv_cov1 + mahalanobis + log_det_cov1_inv_cov2 - d)) 
  if cov_type == 'standard_prior':
    return 1./2 * np.sum(np.diagonal(cov1, axis1=-1, axis2=-2) + mean1**2 - np.log(np.diagonal(cov1, axis1=-1, axis2=-2)) - 1.)      
  
if __name__ == '__main__':  
  test_case = 3

  if test_case == 0:
    # Test KL-divergence
    p = np.array([0.1, 0.4, 0.5])
    q = np.array([0.2, 0.6, 0.2])
    print('my KL_divergence: ', KL_divergence(np.log(p), np.log(q)))
    print('scipy KL_divergence: ', np.sum(kl_div(p, q)))
  
    # Test on noisy one-hot vectors
    eps = 0.
    # ``2, 1``, ``3, 2``, ``3, 1``
    image_feats = {'0':np.array([[eps/2., 1.-eps, eps/2.], [1-eps, eps/2., eps/2.]]), '1':np.array([[eps/2., eps/2., 1.-eps], [eps/2., 1.-eps, eps/2.]]), '2':np.array([[eps/2., eps/2., 1.-eps], [1.-eps, eps/2., eps/2.]])}
    #image_feats = {'0':np.array([[1-eps, eps/2., eps/2.], [eps/2., 1.-eps, eps/2.]]), '1':np.array([[eps/2., 1.-eps, eps/2.], [eps/2., eps/2., 1.-eps]]), '2':np.array([[eps/2., eps/2., 1.-eps], [1.-eps, eps/2., eps/2.]])}
    eps = 0.
    # ``1, 2``, ``2, 3``, ``3, 1``
    audio_feats = {'0':np.array([[1-eps, eps/2., eps/2.], [eps/2., 1.-eps, eps/2.]]), '1':np.array([[eps/2., 1.-eps, eps/2.], [eps/2., eps/2., 1.-eps]]), '2':np.array([[eps/2., eps/2., 1.-eps], [1.-eps, eps/2., eps/2.]])}
    np.savez('tiny_v.npz', **image_feats)
    np.savez('tiny_a.npz', **audio_feats)
  
    alignments = None
    model_configs = {'Kmax':3, 'Mmax':1, 'embedding_dim':3, 'has_null':False}
    speechFeatureFile = 'tiny_a.npz'
    imageFeatureFile = 'tiny_v.npz'
    model = ImageAudioGMMWordDiscoverer(speechFeatureFile, imageFeatureFile, model_configs=model_configs, model_name='tiny')
    model.train_using_EM(num_iterations=10)
    print(model.align(image_feats['0'], image_feats['0']))
    print(model.align(image_feats['1'], image_feats['1']))
    print(model.align(image_feats['2'], image_feats['2']))
    print(model.align_i(0))
    print(model.align_i(1))
    print(model.align_i(2))
    model.print_alignment('tiny')
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
    model = ImageAudioGMMWordDiscoverer(speechFeatureFile, imageFeatureFile, model_configs=model_configs, model_name='image_audio')
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
    
    model_configs = {'Kmax':300, 'Mmax':1, 'embedding_dim':120, 'alignments':[segment_alignments], 'segmentations':segmentations}  
    
    model_configs = {'Kmax':300, 'Mmax':1, 'embedding_dim':120}  
    model = ImageAudioWordDiscoverer(speech_feature_file, image_feature_file, model_configs=model_configs, model_name=exp_dir+'image_audio')
    model.train_using_EM(num_iterations=10)
  elif test_case == 3:
    align_file = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json'
    segment_file = '../data/flickr30k/audio_level/flickr30k_gold_landmarks_mfcc.npz'
    #speech_feature_file = '../data/flickr30k/sensory_level/flickr_concept_kamper_embeddings.npz'
    speech_feature_file = '../data/mscoco/mscoco_kamper_embeddings.npz'
    #image_feature_file = '../data/flickr30k/sensory_level/flickr30k_vgg_penult.npz'
    image_feature_file = '../data/mscoco/mscoco_vgg_penult.npz'
    exp_dir = ''
    # XXX
    #exp_dir = 'exp/nov_10_mscoco_mfcc/'

    # TODO
    # XXX
    model_configs = {'Kmax':10, 'Mmax':3, 'embedding_dim':140, 'has_null':False}  
    model = ImageAudioGMMWordDiscoverer(speech_feature_file, image_feature_file, model_configs=model_configs, model_name=exp_dir+'image_audio')
    model.train_using_EM(num_iterations=10)
