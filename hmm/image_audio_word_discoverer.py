import numpy as np
import math
import time
from copy import deepcopy
from scipy.misc import logsumexp 
from scipy.special import digamma
from sklearn.cluster import KMeans

class ImageAudioWordDiscoverer:
  def __init__(self, speech_feature_file, image_feature_file, model_configs, model_name='image_audio_word_discoverer'):
    self.model_name = model_name
    self.a_corpus = []
    self.v_corpus = []
    
    self.read_corpus(speech_feature_file, image_feature_file)

    # Prior parameters
    self.Kmax = model_configs.get('Kmax', 1000) # Maximum number of vocabs
    self.Mmax = model_configs.get('Mmax', 5) # Maximum number of mixtures per word
    self.embed_dim = model_configs.get('embedding_dim', 130)
    
    # TODO: Match dimensions
    self.g_sb0 = model_configs.get('gamma_sb', np.tile(np.array([1., 1.]), (self.Kmax, 1))) # Stick-breaking beta distribution parameter
    self.g_a0 = model_configs.get('gamma_a', 0.1)
    self.g_at0 = model_configs.get('gamma_a_trans', 0.1) # Alignment transition Dirichlet priors parameter
    self.g_ma0 = model_configs.get('gamma_ma', 0.1) # Audio mixture weight prior parameter
    self.g_mv0 = model_configs.get('gamma_mv', 0.1) # Visual mixture weight prior parameter
    self.k_a0 = model_configs.get('k_a', 0.1) # Ratio between prior variance and obs variance
    self.k_v0 = model_configs.get('k_v', 0.1) # Ratio between prior variance and obs variance
    self.estimate_prior_mean_var()

    self.len_probs = {}
    self.concept_prior = None 
    self.align_init = {}
    self.align_trans = {}
    self.audio_obs_model = {'weights': None,
                            'means': None,
                            'variance': None,
                            'n_mixtures': None
                            }
    self.image_obs_model = {'weights': None,
                            'means': None,
                            'n_mixures': None
                            }    
    
  def read_corpus(self, speech_feat_file, image_feat_file):
    a_npz = np.load(speech_feat_file)
    v_npz = np.load(image_feat_file)
    self.a_corpus = [a_npz[k] for k in sorted(a_npz, key=lambda x:int(x.split()[-1]))]
    self.v_corpus = [v_npz[k] for k in sorted(v_npz, key=lambda x:int(x.split()[-1]))] 
    self.image_feat_dim = len(self.v_corpus[0][0])  
    self.v_corpus = [[np.zeros((self.image_feat_dim,))] + v_npz[k] for k in sorted(v_npz, key=lambda x:int(x.split()[-1]))]  
    
    assert len(self.v_corpus) == len(self.a_corpus)

  def estimate_prior_mean_var(self):
    n_frames_a, n_frames_v = 0., 0.
    self.mu_a0, self.mu_v0 = np.zeros((self.embed_dim,)), np.zeros((self.image_feat_dim,))
    for afeat, vfeat in zip(self.a_corpus, self.v_corpus):
      self.mu_a0 += np.sum(afeat, axis=0)
      self.mu_v0 += np.sum(vfeat, axis=0)
      n_frames_a += afeat.shape[0]
      n_frames_v += vfeat.shape[0]

    self.mu_a0 /= n_frames_a
    self.mu_v0 /= n_frames_v

    self.fixed_variance_a = 0. # XXX: Assume fixed variances  
    self.fixed_variance_v = 0. # XXX: Assume fixed variances  

    c = 3.
    for afeat, vfeat in zip(self.a_corpus, self.v_corpus):
      self.fixed_variance_a += np.sum((afeat - self.mu_a0)**2) / (c * n_frames_a * self.embed_dim)
      self.fixed_variance_v += np.sum((vfeat - self.mu_v0)**2) / (c * n_frames_v * self.image_feat_dim)
    
    # XXX: Check if the means and variance look reasonable
    print('self.mu_a0, self.mu_v0: ', self.mu_a0, self.mu_v0)
    print('Var(a), Var(v): ', self.fixed_variance_a, self.fixed_variance_v) 
         
  def initialize_model(self):
    begin_time = time.time()
    self.compute_translation_length_probabilities()

    for lv in self.len_probs:
      self.align_init[lv] = np.log(1./lv) * np.ones((lv,))

    for lv in self.len_probs:
      self.align_trans[lv] = np.log(1./lv) * np.ones((lv, lv))
    
    self.vs = 1./2 * np.ones((self.Kmax,)) 
    self.vs[-1] = 1.
    # XXX: Check sums to one
    self.concept_prior = np.log(compute_stick_break_prior(self.vs)) 
    print('sum(self.concept_prior): ', sum(np.exp(self.concept_prior)))

    # TODO: Initialize the model
    self.audio_obs_model['weights'] = np.log(1./self.Mmax) * np.ones((self.Kmax, self.Mmax)) 
    self.audio_obs_model['n_mixtures'] = self.Mmax * np.ones((self.Kmax,))
    self.image_obs_model['weights'] = np.log(1./self.Mmax) * np.ones((self.Kmax, self.Mmax))
    self.image_obs_model['n_mixtures'] = self.Mmax * np.ones((self.Kmax,))
    self.audio_obs_model['means'], self.image_obs_model['means'] = self.initialize_mixture_means()
   
    # Initialize hyperparameters for the approximate parameter posteriors
    self.g_aN = {l: self.g_a0 / float(l) * np.ones((l,)) for l in self.len_probs}  # size-L dict of Nv-d array
    self.g_atN = {l: self.g_at0 / float(l) * np.ones((l, l)) for l in self.len_probs} # size-L dict of Nv x Nv matrices 
    self.g_sbN = deepcopy(self.g_sb0) # K-d array
    self.g_maN = self.g_ma0 * np.ones((self.Kmax, self.Mmax)) # Ma-d array
    self.g_mvN = self.g_mv0 * np.ones((self.Kmax, self.Mmax)) # Mv-d array 
    self.k_aN = self.k_a0 * np.ones((self.Kmax, self.Mmax)) 
    self.k_vN = self.k_v0 * np.ones((self.Kmax, self.Mmax)) 

    # Initialize the approximate posteriors
    self.counts_concept = [np.array([deepcopy(self.concept_prior) for _ in range(vfeat.shape[0])]) for vfeat in self.v_corpus]
    self.counts_audio_mixture = None 
    self.counts_image_mixture = None 
    self.counts_align = None 
    self.counts_align_trans = None
     
    # Initialize the expressions involving the digamma function
    self.update_digamma_functions()
    print('takes %.5f s to finish initialzation' % (time.time() - begin_time))

  def initialize_mixture_means(self):
    kmeans_a = KMeans(n_clusters=self.Mmax * self.Kmax, max_iter=3).fit(np.concatenate(self.a_corpus, axis=0))
    kmeans_v = KMeans(n_clusters=self.Kmax, max_iter=3).fit(np.concatenate(self.v_corpus, axis=0))
    return kmeans_a.cluster_centers_.reshape((self.Kmax, self.Mmax, -1)), kmeans_v.cluster_centers_.reshape((self.Kmax, self.Mmax, -1))

  def forward(self, obs_probs):    
    T = obs_probs.shape[0]
    n_state = obs_probs.shape[1]
    forward_probs = -np.inf * np.ones((T, n_state))
    forward_probs[0] = self.digammas_init[n_state] * obs_probs[0] 
    
    for t in range(1, T):
      for j in range(n_state):
        forward_probs[t, j] = logsumexp(forward_probs[t-1] + self.digammas_trans[n_state][:, j]) + obs_probs[t, j]
    
    assert not np.isnan(forward_probs).any() 
    return forward_probs

  def backward(self, obs_probs):
    T = obs_probs.shape[0]
    n_state = obs_probs.shape[1]
    backward_probs = -np.inf * np.ones((T, n_state))
    backward_probs[T-1] = 0.

    for t in range(T-1, 0, -1):
      for j in range(n_state):
        backward_probs[t-1, j] = logsumexp(self.digammas_trans[n_state][j, :] + obs_probs[t, :] + backward_probs[t])
    
    assert not np.isnan(backward_probs).any()
    return backward_probs

  # TODO: Compute concept counts
  def update_concept_counts(self, counts_audio_mixture, counts_image_mixture, counts_align_init, probs_afeat_given_zm, probs_vfeat_given_zm):
    T = len(probs_afeat_given_zm)
    n_state = len(probs_vfeat_given_zm)
    counts_concept = -np.inf * np.ones((n_state, self.Kmax))
    
    probs_afeat_given_z = []
    for i in range(n_state):
      counts_concept[i] = deepcopy(self.digammas_concept)
    
    probs_afeat_given_z = logsumexp(np.exp(counts_audio_mixture) * probs_afeat_given_zm, axis=-1)
    counts_concept += counts_align_init.T @ np.array(probs_afeat_given_z) 
    counts_concept += logsumexp(np.exp(counts_image_mixture) * probs_vfeat_given_zm, axis=-1)
 
    # Normalize
    for i in range(n_state):
      counts_concept[i] -= logsumexp(counts_concept[i])

    return counts_concept

  def update_audio_mixture_counts(self, probs_afeat_given_zm):
    # XXX
    counts_audio_mixture = self.digammas_ma + probs_afeat_given_zm 
    # Normalize
    norm_factors = logsumexp(counts_audio_mixture, axis=-1) 
    counts_audio_mixture = np.transpose(np.transpose(counts_audio_mixture, (2, 0, 1)) - norm_factors, (1, 2, 0)) 
    print('sum(counts_audio_mixture): ', np.sum(np.exp(counts_audio_mixture), axis=-1))
    return counts_audio_mixture 

  def update_image_mixture_counts(self, probs_vfeat_given_zm):
    counts_image_mixture = self.digammas_mv + probs_vfeat_given_zm
    # Normalize
    norm_factors = logsumexp(counts_image_mixture, axis=-1) 
    counts_image_mixture = np.transpose(np.transpose(counts_image_mixture, (2, 0, 1)) - norm_factors, (1, 2, 0)) 
    print('sum(counts_image_mixture): ', np.sum(np.exp(counts_image_mixture), axis=-1))
    return counts_image_mixture 

  def update_alignment_transition_counts(self, forward_probs, backward_probs, obs_probs):  
    T = len(forward_probs)
    n_state = len(forward_probs)
    counts_align_trans = -np.inf * np.ones((n_state, n_state))
    prob_ij_given_obs = -np.inf * np.ones((T, n_state, n_state))
    counts_jump = {}
    
    for t in range(T-1):
      prob_ij_given_obs[t] = (forward_probs[t] + self.digammas_trans[n_state].T).T + obs_probs[t+1] + backward_probs[t+1] 
      for s in range(n_state):
        prob_ij_given_obs[t, s] -= logsumexp(prob_ij_given_obs[t])

      for s in range(n_state):
        for next_s in range(n_state):
          jump = next_s - s
          if jump not in counts_jump:
            counts_jump[jump] = [prob_ij_given_obs[s, next_s]]
          else:
            counts_jump[jump].append(prob_ij_given_obs[s, next_s])  

    for s in range(n_state):
      for next_s in range(n_state):
        jump = next_s - s   
        counts_align_trans[s, next_s] = logsumexp(np.asarray(counts_jump[jump]))
    
    print('sum(counts_align_trans): ', np.sum(np.exp(counts_align_trans), axis=-1))
    assert not np.isnan(counts_align_trans).any()
    return counts_align_trans
           
  def update_alignment_initial_counts(self, forward_probs, backward_probs):
    T = len(forward_probs)
    n_state = forward_probs.shape[1]
    prob_i_given_obs = forward_probs + backward_probs 
    for t in range(T):
      norm_factor = logsumexp(prob_i_given_obs[t])
      prob_i_given_obs[t] = prob_i_given_obs[t] - norm_factor
      
    counts_align = -np.inf * np.ones((n_state,))
    for i in range(n_state):
      counts_align[i] = logsumexp(prob_i_given_obs[:, i]) 

    print('sum(counts_align_init): ', np.sum(np.exp(counts_align), axis=-1))
    assert not np.isnan(counts_align).any()
    return counts_align 
 
  # TODO: Match dimensions
  def update_posterior_hyperparameters(self):
    new_g_maN = self.g_ma0 / self.Mmax * np.ones((self.Kmax, self.Mmax))
    new_g_mvN = self.g_mv0 / self.Mmax * np.ones((self.Kmax, self.Mmax))
    new_g_aN = {l: self.g_a0 / l * np.ones((l,)) for l in self.len_probs}
    new_g_atN = {l: self.g_at0 / l * np.ones((l, l)) for l in self.len_probs}
    new_g_sbN = deepcopy(self.g_sb0) 
    new_k_aN = self.k_a0 * np.ones((self.Kmax, self.Mmax))
    new_k_vN = self.k_v0 * np.ones((self.Kmax, self.Mmax))

    aggregated_counts = np.array([logsumexp(counts, axis=0) for counts in self.counts_audio_mixture])
    new_g_maN += logsumexp(aggregated_counts, axis=0)
    new_k_aN += logsumexp(aggregated_counts, axis=0)

    aggregated_counts = None
    aggregated_counts = np.array([logsumexp(counts, axis=0) for counts in self.counts_image_mixture])
    new_g_mvN += logsumexp(aggregated_counts, axis=0) 
    new_k_vN += logsumexp(aggregated_counts, axis=0)
    
    for n_states in self.len_probs:
      new_g_aN[n_states] += logsumexp(np.array(self.counts_align[n_states]), axis=0)
      new_g_atN[n_states] += logsumexp(np.array(self.counts_align_trans[n_states]), axis=0)  

    aggregated_counts, aggregated_counts_gt = [], []
    for counts in self.counts_concept:
      n_states = counts.shape[0]
      counts_gt = -np.inf * np.ones(counts.shape)
      for k in range(self.Kmax-1):
        counts_gt[:, k] = logsumexp(counts[:, k+1:]) 
      aggregated_counts.append(logsumexp(counts, axis=0)) 
      aggregated_counts_gt.append(logsumexp(counts_gt, axis=0))
    new_g_sbN[:, 0] += logsumexp(np.array(aggregated_counts), axis=0)
    new_g_sbN[:, 1] += logsumexp(np.array(aggregated_counts_gt), axis=0)
    
    self.g_maN = new_g_maN
    self.g_mvN = new_g_mvN
    self.g_aN = new_g_aN
    self.g_atN = new_g_atN
    self.g_sbN = new_g_sbN
    self.k_aN = new_k_aN 
    self.k_vN = new_k_vN

  def update_digamma_functions(self):
    self.digammas_init = {l:None for l in self.g_aN}
    self.digammas_trans = {l:None for l in self.g_atN}
    
    for l in self.g_aN:
      self.digammas_init[l] = digamma(np.exp(self.g_aN[l])) - digamma(np.exp(logsumexp(self.g_aN[l])))
      self.digammas_trans[l] = (digamma(np.exp(self.g_atN[l])).T - digamma(np.exp(logsumexp(self.g_aN[l])))).T
    
    self.digammas_concept = np.zeros((self.Kmax))
    self.digammas_ma = np.zeros((self.Kmax, self.Mmax))
    self.digammas_mv = np.zeros((self.Kmax, self.Mmax))
    
    for k in range(self.Kmax):
      for j in range(k+1, self.Kmax):
        self.digammas_concept[k] += digamma(np.exp(self.g_sbN[j, 1])) - digamma(np.exp(logsumexp(self.g_sbN[j])))
      self.digammas_concept[k] += digamma(np.exp(self.g_sbN[k, 0])) - digamma(np.exp(logsumexp(self.g_sbN[k])))
      
      for m in range(self.Mmax):
        self.digammas_ma[k] = digamma(np.exp(self.g_maN[k])) - digamma(np.exp(logsumexp(self.g_maN[k])))
      
      for m in range(self.Mmax):
        self.digammas_mv[k] = digamma(np.exp(self.g_mvN[k])) - digamma(np.exp(logsumexp(self.g_mvN[k])))
         
  def update_concept_alignment_model(self):
    vs = np.zeros((self.Kmax,)) 
    vs = np.exp(self.g_sbN[:, 0]) / (np.exp(self.g_sbN[:, 0]) + np.exp(self.g_sbN[:, 1]))
    
    self.concept_prior = None
    self.concept_prior = np.log(compute_stick_break_prior(vs))
    
    for l in self.g_aN:
      self.align_init[l] = self.g_aN[l] - logsumexp(self.g_aN[l])

    for l in self.g_atN:
      self.align_trans[l] = (self.g_atN[l].T - logsumexp(self.g_atN[l], axis=1)).T

  def update_observation_models(self):
    self.audio_obs_model['weights'] = -np.inf * np.ones(self.g_maN.shape)
    self.image_obs_model['weights'] = -np.inf * np.ones(self.g_mvN.shape)
    exp_audio_mixture_means = np.tile(self.k_a0 * self.mu_a0, (self.audio_obs_model['means'].shape[0], self.audio_obs_model['means'].shape[1], 1))
    exp_image_mixture_means = np.tile(self.k_v0 * self.mu_v0, (self.image_obs_model['means'].shape[0], self.image_obs_model['means'].shape[1], 1))
    exp_num_audio_mixture = np.zeros((self.Kmax, self.Mmax))
    exp_num_image_mixture = np.zeros((self.Kmax, self.Mmax))

    for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
      for k in range(self.Kmax):
        # TODO: Change add to logsumexp
        exp_audio_mixture_means[k] += np.exp(self.counts_audio_mixture[ex][:, k, :]).T @ afeats
        exp_image_mixture_means[k] += np.exp(self.counts_image_mixture[ex][:, k, :]).T @ vfeats

        exp_num_audio_mixture[k] += np.sum(np.exp(self.counts_audio_mixture[ex][:, k, :]), axis=0)
        exp_num_image_mixture[k] += np.sum(np.exp(self.counts_image_mixture[ex][:, k, :]), axis=0)

    self.audio_obs_model['means'] = exp_audio_mixture_means / (self.k_a0 + exp_num_audio_mixture)
    self.image_obs_model['means'] = exp_image_mixture_means / (self.k_a0 + exp_num_image_mixture)

    # Normalize
    self.audio_obs_model['weights'] = (self.g_maN.T - logsumexp(self.g_maN, axis=1)).T
    self.image_obs_model['weights'] = (self.g_mvN.T - logsumexp(self.g_mvN, axis=1)).T 
    
    print('sum(p(m_at|z_t)): ', np.sum(np.exp(self.audio_obs_model['weights']), axis=-1))
    print('sum(p(m_vt|z_t)): ', np.sum(np.exp(self.image_obs_model['weights']), axis=-1))


  def train_using_EM(self, num_iterations=10, write_model=True):
    self.initialize_model()
    #if write_model:
    #  self.print_model('initial_model.txt') 
     
    print("Initial log likelihood: ", self.compute_log_likelihood()) 
    for n in range(num_iterations):
      begin_time = time.time()
      new_counts_concept = [] # length-Nd list of Nv x K x Ma x Mv array
      new_counts_audio_mixture, new_counts_image_mixture = [], []
      new_counts_align = {l: [] for l in self.len_probs}
      new_counts_align_trans = {l: [] for l in self.len_probs}

      for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
        log_probs_a_given_zm = self.log_probs_afeat_given_z_m(afeats)
        log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)
        counts_audio_mixture = self.update_audio_mixture_counts(log_probs_a_given_zm)
        counts_image_mixture = self.update_image_mixture_counts(log_probs_v_given_zm)
        new_counts_audio_mixture.append(counts_audio_mixture)
        new_counts_image_mixture.append(counts_image_mixture)

        log_probs_av_given_i = self.log_probs_afeat_vfeat_given_i(log_probs_a_given_zm, log_probs_v_given_zm, self.counts_concept[ex], counts_audio_mixture, counts_audio_mixture)
        
        forward_probs = self.forward(log_probs_av_given_i)
        backward_probs = self.backward(log_probs_av_given_i)
        # Estimate counts
        # TODO: match inputs
        counts_align = self.update_alignment_initial_counts(forward_probs, backward_probs)
        counts_align_trans = self.update_alignment_transition_counts(forward_probs, backward_probs, log_probs_av_given_i)
        new_counts_align[len(vfeats)].append(counts_align)
        new_counts_align_trans[len(vfeats)].append(counts_align_trans)
        new_counts_concept.append(self.update_concept_counts(counts_audio_mixture, counts_image_mixture, counts_align, log_probs_a_given_zm, log_probs_v_given_zm))
     
      print('Take %.5f for E step' % (time.time() - begin_time)) 
      self.counts_concept = deepcopy(new_counts_concept)
      self.counts_audio_mixture = deepcopy(new_counts_audio_mixture)
      self.counts_image_mixture = deepcopy(new_counts_image_mixture)
      self.counts_align = deepcopy(new_counts_align)
      self.counts_align_trans = deepcopy(new_counts_align_trans)

      # Update posterior hyperparameters
      self.update_posterior_hyperparameters()

      # Update parameters
      self.update_concept_alignment_model()
      self.update_observation_models()
      print('Take %.5f for M step' % (time.time() - begin_time)) 
      
      print('Log likelihood after iteration %d: %.5f' % (n, self.compute_log_likelihood()))
  
  # Compute log likelihood using the formula: \log p(a, v) = \sum_i=1^n \log p(v_i) + \log \sum_{A} p(A|v) * p(x|A, v)
  def compute_log_likelihood(self):
    ll = 0.
    for ex, (afeats, vfeats) in enumerate(zip(self.a_corpus, self.v_corpus)):
      log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)
      log_probs_a_given_zm = self.log_probs_afeat_given_z_m(afeats)
      log_probs_v_given_z = logsumexp(log_probs_v_given_zm + self.audio_obs_model['weights'], axis=-1)
      log_probs_v = logsumexp(self.concept_prior + log_probs_v_given_z, axis=1)
      # TODO: Check sum = 1  
      log_probs_z_given_v = ((self.concept_prior + log_probs_v_given_z).T - log_probs_v).T
      print('sum(p(z)): ', np.sum(np.exp(self.concept_prior)))
      print('sum(p(z|v)): ', np.sum(np.exp(log_probs_z_given_v), axis=1))
      log_prob_v_tot = np.sum(log_probs_v)
      log_probs_a_i_given_v = self.log_prob_afeats_align_given_vfeats(log_probs_a_given_zm, log_probs_z_given_v)
      log_prob_a_given_v_tot = logsumexp(log_probs_a_i_given_v[-1])
      ll += log_prob_v_tot + log_prob_a_given_v_tot
    return ll
    
  # TODO: Match inputs
  def log_prob_afeats_align_given_vfeats(self, log_probs_a_given_zm, log_probs_z_given_v):
    T = len(log_probs_a_given_zm)
    n_states = len(log_probs_z_given_v)
    log_probs_a_i_given_v = -np.inf * np.ones((T, n_states)) 
    log_probs_a_given_z = logsumexp(self.audio_obs_model['weights'] + log_probs_a_given_zm[0], axis=-1)
    log_probs_a_given_v_i = logsumexp(log_probs_z_given_v + log_probs_a_given_z, axis=-1)
    log_probs_a_i_given_v[0] = self.align_init[n_states] + log_probs_a_given_v_i
    for t in range(1, T):
      log_probs_a_given_z_t = logsumexp(self.audio_obs_model['weights'] + log_probs_a_given_zm[t], axis=-1)
      log_probs_a_given_v_t = logsumexp(log_probs_z_given_v + log_probs_a_given_z_t, axis=-1)
      log_probs_a_i_given_v[t] = logsumexp(log_probs_a_i_given_v[t-1] + self.align_trans[n_states].T, axis=-1) + log_probs_a_given_v_t 
    
    return log_probs_a_i_given_v
    
  # TODO
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

  # Compute approximate p(x_t|i(t))
  #def log_prob_afeat_given_i(self, afeat, i): 
  # for t in r
      
  def log_prob_afeat_given_z_m(self, afeat, k, m):
    return gaussian(afeat, self.audio_obs_model['means'][k, m], self.fixed_variance_a * np.ones((self.embed_dim, self.embed_dim)), log_prob=True)

  def log_prob_vfeat_given_z_m(self, vfeat, k, m):
    return gaussian(vfeat, self.image_obs_model['means'][k, m], self.fixed_variance_v * np.ones((self.image_feat_dim, self.image_feat_dim)), log_prob=True)

  #def log_prob_afeat_given_z_m(self, afeat):
  # Compute approximate p(afeat_t|i) for all t and i and store them in a T x N(v) matrix 
  # XXX: Match arguments
  def log_probs_afeat_given_i(self, log_probs_a_given_zm, counts_concept, counts_audio_mixture):
    T = log_probs_a_given_zm.shape[0]
    n_states = counts_concept.shape[0] 
    log_probs_a_given_i = -np.inf*np.ones((T, n_states))

    for t in range(T):
      log_probs_a_m_given_v_t = counts_concept @ (counts_audio_mixture[t] * log_probs_a_given_zm[t])
      log_probs_a_given_i[t, :] = logsumexp(log_probs_a_m_given_v_t, axis=1)

    return log_probs_a_given_i
  
  # Compute approximate p(afeat_t|z_i(t), m_t) for all t, z_i(t) and m and store them in a T x K x M matrix
  def log_probs_afeat_given_z_m(self, afeats):
    log_probs = []
    T = len(afeats)
    for k in range(self.Kmax):
      log_probs_m = []
      for m in range(self.Mmax):
        log_probs_m.append(self.log_prob_afeat_given_z_m(afeats, k, m))
      log_probs.append(log_probs_m)
    return np.transpose(np.asarray(log_probs), [2, 0, 1])
  
  def log_probs_vfeat_given_z_m(self, vfeats):
    log_probs = []
    for k in range(self.Kmax):
      log_probs_m = []
      for m in range(self.Mmax):
        log_prob_m = self.log_prob_vfeat_given_z_m(vfeats, k, m)
        log_probs_m.append(log_prob_m)
       
      log_probs.append(log_probs_m)
    return np.transpose(np.asarray(log_probs), [2, 0, 1])
 
  # Compute approximate p(afeat_t, vfeat_i|i) for all t and i and store them in a T x N(v) matrix 
  def log_probs_afeat_vfeat_given_i(self, log_probs_a_given_zm, log_probs_v_given_zm, counts_concept, counts_audio_mixture, counts_image_mixture):
    log_probs = []
    # XXX: match arguments
    log_probs_a_given_i = self.log_probs_afeat_given_i(log_probs_a_given_zm, counts_concept, counts_audio_mixture)
    log_probs_v = logsumexp(log_probs_v_given_zm, axis=(1, 2))
    return log_probs_a_given_i + log_probs_v
  
  def align(self, afeats, vfeats, decode_method='viterbi'):
    T = len(afeats)
    n_states = len(vfeats)   
    scores = np.zeros((n_states,))
    back_ptrs = np.zeros((T, n_states), dtype=int)
    align_probs = []
    log_probs_a_given_zm = self.log_probs_afeat_given_z_m(afeats)
    log_probs_v_given_zm = self.log_probs_vfeat_given_z_m(vfeats)
    log_probs_v_given_z = logsumexp(log_probs_v_given_zm + self.audio_obs_model['weights'], axis=-1)  
    log_probs_v = logsumexp(self.concept_prior + log_probs_v_given_z, axis=1)  
    log_probs_z_given_v = ((self.concept_prior + log_probs_v_given_z).T - log_probs_v).T
    
    log_probs_a_given_z = logsumexp(self.audio_obs_model['weights'] + log_probs_a_given_zm[0], axis=-1)
    scores = self.align_init[n_states] + logsumexp(log_probs_z_given_v + log_probs_a_given_z, axis=-1)
    for t in range(1, T): 
      log_probs_a_given_z_t = logsumexp(self.audio_obs_model['weights'] + log_probs_a_given_zm[t], axis=-1)
      log_probs_a_given_v = logsumexp(log_probs_z_given_v + log_probs_a_given_z_t, axis=-1) 
      score_candidates = (scores[np.newaxis, :] + self.align_trans[n_states].T).T + log_probs_a_given_v
      back_ptrs[t] = np.argmax(score_candidates, axis=0)
      scores = np.max(score_candidates, axis=0)
      align_probs.append(scores.tolist())
      
    # TODO: Find the optimal word indices as well
    cur_state = np.argmax(scores)
    best_path = [int(cur_state)]
    for t in range(T-1, 0, -1):
      cur_state = back_ptrs[t, cur_state]
      best_path.append(int(cur_state))
      
    return best_path[::-1], align_probs 
  
  # TODO:
  #def print_model(self):
  #def print_alignment(self, file_prefix):

def compute_stick_break_prior(vs):
  K = len(vs)
  pvs = np.cumprod(1-vs)
  prior = np.zeros((K,))
  prior[0] = vs[0]
  prior[1:] = pvs[:-1] * vs[1:]
  return prior

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

if __name__ == '__main__':
  eps = 0.
  image_feats = {'0_1':np.array([[eps/2., 1.-eps, eps/2.], [1-eps, eps/2., eps/2.]]), '0_2':np.array([[eps/2., eps/2., 1.-eps], [eps/2., 1.-eps, eps/2.]]), '0_3':np.array([[eps/2., eps/2., 1.-eps], [1.-eps, eps/2., eps/2.]])}
  eps = 0.1
  audio_feats = {'0_1':np.array([[1-eps, eps/2., eps/2.], [eps/2., 1.-eps, eps/2.]]), '0_2':np.array([[eps/2., 1.-eps, eps/2.], [eps/2., eps/2., 1.-eps]]), '0_3':np.array([[eps/2., eps/2., 1.-eps], [1.-eps, eps/2., eps/2.]])}
  
  np.savez('tiny_v.npz', **image_feats)
  np.savez('tiny_a.npz', **audio_feats)
  
  model_configs = {'Kmax':3, 'Mmax':1, 'embedding_dim':3}  
  speechFeatureFile = 'tiny_a.npz'
  imageFeatureFile = 'tiny_v.npz'
  model = ImageAudioWordDiscoverer(speechFeatureFile, imageFeatureFile, model_configs=model_configs, model_name='image_audio')
  model.train_using_EM(num_iterations=10)
  print(model.align(image_feats['0_1'], image_feats['0_1']))
  print(model.align(image_feats['0_2'], image_feats['0_2']))
  print(model.align(image_feats['0_2'], image_feats['0_2']))
