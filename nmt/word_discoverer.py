from __future__ import division, generators

import dynet as dy
import numpy as np
import length_normalization
import decorators
import batcher
import six
import plot
import os
import json
import time

from vocab import Vocab
from serializer import Serializable, DependentInitParam
from search_strategy import BeamSearch
from embedder import SimpleWordEmbedder
from decoder import MlpSoftmaxDecoder
from translator import Translator
from output import TextOutput
from model import HierarchicalModel, GeneratorModel
from reports import HTMLReportable
from decorators import recursive_assign, recursive

# Reporting purposes
from lxml import etree


DEBUG = True
class WordDiscoverer(Translator, Serializable, HTMLReportable):
  '''
  A default translator based on attentional sequence-to-sequence models.
  '''

  yaml_tag = u'!WordDiscoverer'

  def __init__(self, src_embedder, encoder, attender, trg_embedder, trg_data, decoder):
    '''Constructor.

    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_embedder: A word embedder for the output language
    :param trg_data: A database with all the image concepts to be aligned
    :param decoder: A decoder
    '''
    super(WordDiscoverer, self).__init__()
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.trg_data = trg_data # XXX: Need a better name; a hack to allow access to target sequence since we only try to learn the alignment
    self.decoder = decoder

    self.register_hier_child(self.encoder)
    self.register_hier_child(self.decoder)

  def shared_params(self):
    return [set(["src_embedder.emb_dim", "encoder.input_dim"]),
            # TODO: encoder.hidden_dim may not always exist (e.g. for CNN encoders), need to deal with that case
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]),
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"])]
 
  def dependent_init_params(self):
    return [DependentInitParam(param_descr="src_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].src_reader.vocab_size()),
            DependentInitParam(param_descr="decoder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size())]

  def initialize(self, args):
      # Search Strategy
    len_norm_type   = getattr(length_normalization, args.len_norm_type)
    self.search_strategy = BeamSearch(b=args.beam, max_len=args.max_len, len_norm=len_norm_type(**args.len_norm_params))
    self.report_path = args.report_path

  def calc_loss(self, src, trg, info=None):
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    self.attender.start_sent(encodings)
    self.decoder.initialize()
    self.decoder.add_input(self.trg_embedder.embed(0))  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not batcher.is_batched(src):
      for ref_word in trg:
        self.attender.calc_attention(self.trg_embedder.embed(ref_word))
      
      contexts = self.attender.calc_context(self.decoder.state.output()) # XXX: The attender does not actually use the state; keep it only for consistency of attender definition
      for i, ref_word in enumerate(trg):
        word_loss = self.decoder.calc_loss(dy.select_cols(contexts, [i]), ref_word)
        losses.append(word_loss)
        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    # minibatch mode
    else:
      max_len = max([len(single_trg) for single_trg in trg])

      for i in range(max_len):
        ref_word = batcher.mark_as_batch([single_trg[i] if i < len(single_trg) else Vocab.ES for single_trg in trg])
        self.attender.calc_attention(self.trg_embedder.embed(ref_word)) 

      contexts = self.attender.calc_context(self.decoder.state.output()) # XXX: The attender does not actually use the state; keep it only for consistency of attender definition
      #if DEBUG:
      #  print('Contexts dim(): ', contexts.dim())      
      for i in range(max_len):
        ref_word = batcher.mark_as_batch([single_trg[i] if i < len(single_trg) else Vocab.ES for single_trg in trg])
        word_loss = self.decoder.calc_loss(dy.pick(contexts, i, dim=1), ref_word)
        mask_exp = dy.inputVector([1 if i < len(single_trg) else 0 for single_trg in trg])
        mask_exp = dy.reshape(mask_exp, (1,), len(trg))
        word_loss = word_loss * mask_exp
        losses.append(word_loss)

        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    return dy.esum(losses)

  def generate(self, src, idx):
    # Not including this as a default argument is a hack to get our documentation pipeline working
    if DEBUG:
      print('src, trg_data: ', src, self.trg_data.data[idx])
    search_strategy = self.search_strategy
    if search_strategy == None:
      search_strategy = BeamSearch(1, len_norm=NoNormalization())
    if not batcher.is_batched(src):
      src = batcher.mark_as_batch([src])
    
    trg_sent = self.trg_data.data[idx] 
    outputs = []
    for src_sent in src:
      if DEBUG:
        print('src len: ', len(src))
        print('src_sent len, trg_sent len: ', len(src_sent), len(trg_sent))
      src_embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder.transduce(src_embeddings)
      self.attender.start_sent(encodings)
      self.decoder.initialize()
      # XXX: Placeholder for consistency of defintion
      trg_embeddings = self.trg_embedder.embed_sent(trg_sent)
        
      for trg_embedding in trg_embeddings:
        self.attender.calc_attention(trg_embedding)
      
      attentions = self.attender.normalize()
      if DEBUG:
        print('encodings size: ', encodings.as_tensor().dim())
        print('attention size: ', attentions.dim())
        print('src vocab: ', self.src_vocab[:10], self.src_vocab[0])
      #output_actions = search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, src_length=len(src_sent))
      # Append output to the outputs
      alignment = np.argmax(attentions.npvalue(), axis=1).tolist()
      output_actions = [str(trg_idx) for trg_idx in alignment] 
    
      # In case of reporting  
      start_time = time.time()
      if self.report_path is not None:
        src_words = [self.src_vocab[w] for w in src_sent]
        trg_words = [self.trg_vocab[w] for w in trg_sent]
        # TODO: Double check this
        self.set_html_input(idx, src_words, trg_words, attentions)
        self.set_html_path('{}.{}'.format(self.report_path, str(idx)))
      
        data_info = []
        if os.path.isfile(self.report_path + '.json'):
          with open(self.report_path + '.json', 'r') as f:
            data_info = json.load(f)

        sent_info = {'index': idx,
                     'src_sent': src_words,
                     'trg_sent': trg_words,
                     'attentions': attentions.npvalue().tolist(),
                     'alignment': alignment}
        
        data_info.append(sent_info)
        with open(self.report_path + '.json', 'w') as f:
          json.dump(data_info, f, indent=4, sort_keys=True)
      print('report takes %s to generate', str(start_time - time.time()))
      # XXX: A hack to allow alignment to be the output
      trg_vocab_dict = {str(i): w for i, w in enumerate(trg_words)}
      outputs.append(TextOutput(output_actions, trg_vocab_dict))
    return outputs
  
  '''def align(self, src, trg):
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.src_encoder.transduce(embeddings)
    self.attender.start_sent(src)

    # Convert the list of trg variable-length sequence to a list of fixed-length ref-word sequence,
    # with each ref-word autobatched
    if not batcher.is_batched(src):
      src = batcher.mark_as_batch([src])

    trg_lens = [len(single_trg) for single_trg in trg]
    maxlen = max(trg_lens)
    
    for i in range(maxlen):
      ref_word = batcher.mark_as_batch([single_trg[i] if i < len(single_trg) else Vocab.ES for single_trg in trg])    
      trg_embedding = self.trg_embedder.embed_sent(ref_word)   
      # Generate the attention vectors for each target words
      self.attender.calc_attention(trg_embedding)
    
    # Normalize each attention vector over the target words
    attentions = self.attender.normalize().values()

    # Save the attention weights
    '''    

  @recursive_assign
  def html_report(self, context=None):
    assert(context is None)
    idx, src, trg, att = self.html_input
    path_to_report = self.html_path
    filename_of_report = os.path.basename(path_to_report)
    html = etree.Element('html')
    head = etree.SubElement(html, 'head')
    title = etree.SubElement(head, 'title')
    body = etree.SubElement(html, 'body')
    report = etree.SubElement(body, 'h1')
    if idx is not None:
      title.text = report.text = 'Translation Report for Sentence %d' % (idx)
    else:
      title.text = report.text = 'Translation Report'
    main_content = etree.SubElement(body, 'div', name='main_content')

    # Generating main content
    captions = [u"Source Words", u"Target Words"]
    inputs = [src, trg]
    for caption, inp in six.moves.zip(captions, inputs):
      if inp is None: continue
      sent = ' '.join(inp)
      p = etree.SubElement(main_content, 'p')
      p.text = u"{}: {}".format(caption, sent)

    # Generating attention
    if not any([src is None, trg is None, att is None]):
      attention = etree.SubElement(main_content, 'p')
      att_text = etree.SubElement(attention, 'b')
      att_text.text = "Attention:"
      etree.SubElement(attention, 'br')
      att_mtr = etree.SubElement(attention, 'img', src="{}.attention.png".format(filename_of_report))
      attention_file = u"{}.attention.png".format(path_to_report)

      if type(att) == dy.Expression:
        attentions = att.npvalue()
      elif type(att) == list:
        attentions = np.concatenate([x.npvalue() for x in att], axis=1)
      elif type(att) != np.ndarray:
        raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(attentions)))
      plot.plot_attention(src, trg, attentions, file_name = attention_file)

    # return the parent context to be used as child context
    return html

