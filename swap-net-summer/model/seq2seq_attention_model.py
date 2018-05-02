
from collections import namedtuple

import numpy as np
import tensorflow as tf

import seq2seq



HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps,sent_enc_timesteps,max_article_sentences '
                     'min_input_len, num_hidden, emb_dim,vocabulary_size, max_grad_norm, '
                     'num_softmax_samples')




class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab, num_gpus=0):
    self.hps = hps
    self.vocab = vocab
    self.num_gpus = num_gpus
    self.cur_gpu = 0
    self.loss_ind=3

  def make_feed(self,batch):
    return {self.enc_batch:batch.enc_batch, self.dec_batch:batch.dec_batch,
                                       self.sent_enc_batch:batch.sent_enc_batch,self.sent_dec_batch:batch.sent_dec_batch,
                                       self.target_batch:batch.target_batch,self.extend_target_batch:batch.extend_target_batch,self.sent_target_batch:batch.sent_target_batch,self.switch_batch:batch.switch_batch,
                                       self.enc_input_lens:batch.enc_input_lens,self.sent_enc_input_lens:batch.sent_enc_input_lens,self.dec_output_lens:batch.dec_output_lens, 
                                       self.word_weights_batch:batch.word_weights,self.switch_weights_batch:batch.switch_weights,self.sent_decwords_batch:batch.sent_decwords,self.words_decsent_batch:batch.words_decsent,
                                       self.weights_sent_decwords_batch:batch.weights_sent_decwords,self.weights_words_decsent_batch:batch.weights_words_decsent}


  def run_train_step(self, sess, batch,loss_ind=0):
    self.loss_ind=loss_ind
    to_return =[self.train_op, self.summaries, self.loss_to_minimize, self.global_step]
    out=[self.decoder_outputs_dists, self.sent_decoder_outputs_dists,self.final_log_dists]
    loss=[self.loss,self.word_loss,self.sent_loss,self.word_loss_null,self.sent_loss_null,self.switch_loss]
    return sess.run([to_return,out,loss],
                    feed_dict=self.make_feed(batch))

  def run_eval_step(self, sess,batch): 
    to_return = [self.summaries, self.total_loss, self.global_step]
  
    out=[self.decoder_outputs, self.sent_decoder_outputs,self.final_log_dists]
    loss=[self.loss,self.word_loss,self.sent_loss,self.word_loss_null,self.sent_loss_null,self.switch_loss]
    return sess.run([to_return,out,loss],feed_dict=self.make_feed(batch))


  def mask_and_avg(self,values,weights):
 
    dec_lens = tf.reduce_sum(weights, axis=1) 
    dec_lens+= 1e-12
    values_per_step = [v * weights[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens 
    return tf.reduce_mean(values_per_ex) 



  def _calc_final_dist(self, vocab_dists, attn_dists):

    with tf.variable_scope('final_distribution'):

      extras=self.hps.sent_enc_timesteps-self.hps.max_article_sentences 
      extended_vsize = self.hps.enc_timesteps + self.hps.sent_enc_timesteps 
      extra_zeros = tf.zeros((self.hps.batch_size, self.hps.sent_enc_timesteps))
      vocab_dists_extended = [tf.concat(1,[dist, extra_zeros]) for dist in vocab_dists] 

  
      batch_nums = tf.range(0, limit=self.hps.batch_size) 
      batch_nums = tf.expand_dims(batch_nums, 1) 
      attn_len = self.hps.sent_enc_timesteps 
      batch_nums = tf.tile(batch_nums, [1, attn_len]) 
      sent_ind=tf.range(self.hps.enc_timesteps, extended_vsize)
      sent_ind=tf.tile(tf.expand_dims(sent_ind,1),[self.hps.batch_size,1])
      indices = tf.stack( (batch_nums,tf.reshape(sent_ind,[self.hps.batch_size,self.hps.sent_enc_timesteps]) ), axis=2)

      shape = [self.hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] 

      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists  


  def get_loss(self,final_dists,targets,weights):
	  log_dists = [tf.log(dist+1e-12) for dist in final_dists]
          loss_per_step = [] 
          batch_nums = tf.range(0, limit=self.hps.batch_size) 
          for dec_step, log_dist in enumerate(log_dists):
            target = targets[dec_step] 
            indices = tf.stack( (batch_nums, target), axis=1) 
            losses = tf.gather_nd(-log_dist, indices) 
            loss_per_step.append(losses)
	  loss=self.mask_and_avg(loss_per_step,weights)
	  return loss


  



  def add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self.hps

    self.enc_batch = tf.placeholder(tf.int32,
                                  [hps.batch_size, hps.enc_timesteps],
                                  name='enc_batch')
    self.dec_batch = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps+1],
                                     name='dec_batch')
    self.sent_enc_batch = tf.placeholder(tf.int32,
                                      [hps.batch_size, hps.sent_enc_timesteps],
                                      name='sent_enc_batch')
    self.sent_dec_batch = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps+1],
                                     name='sent_dec_batch')
    self.target_batch = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='target_batch')
    self.extend_target_batch = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='extend_target_batch')
    self.sent_target_batch = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='sent_target_batch')
    self.switch_batch = tf.placeholder(tf.float32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='switch_batch')
    self.sent_decwords_batch=tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps,hps.max_article_sentences],name='sent_decwords')
    self.words_decsent_batch=tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps,hps.enc_timesteps],name='words_decsent')
    self.weights_sent_decwords_batch=tf.placeholder(tf.float32,
                                   [hps.batch_size, hps.dec_timesteps,hps.max_article_sentences],name='weights_sent_decwords')
    self.weights_words_decsent_batch=tf.placeholder(tf.float32,
                                   [hps.batch_size, hps.dec_timesteps,hps.enc_timesteps],name='weights_words_decsent')

    self.enc_input_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='enc_input_lens')
    self.sent_enc_input_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='sent_enc_input_lens')
    self.dec_output_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                         name='dec_output_lens')
    self.word_weights_batch = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                        name='word_weights_batch')
    self.switch_weights_batch = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                          name='switch_weights_batch')
    


  def add_seq2seq(self):
    hps = self.hps
    vsize = hps.vocabulary_size
    threshold=0.5

    with tf.variable_scope('seq2seq'):
      encoder_inputs = tf.unpack(tf.transpose(self.enc_batch))
      decoder_inputs = tf.unpack(tf.transpose(self.dec_batch))
      sent_encoder_inputs = tf.unpack(tf.transpose(self.sent_enc_batch))
      sent_decoder_inputs = tf.unpack(tf.transpose(self.sent_dec_batch))
      targets = tf.unpack(tf.transpose(self.target_batch))
      extend_targets = tf.unpack(tf.transpose(self.extend_target_batch))
      sent_targets = tf.unpack(tf.transpose(self.sent_target_batch))
      switch = tf.unpack(tf.transpose(self.switch_batch))
      word_weights = tf.unpack(tf.transpose(self.word_weights_batch))
      switch_weights = tf.unpack(tf.transpose(self.switch_weights_batch))
      sent_decwords=tf.unpack(tf.transpose(self.sent_decwords_batch,perm=[1,0,2]))
      words_decsent=tf.unpack(tf.transpose(self.words_decsent_batch,perm=[1,0,2]))
      weights_sent_decwords=tf.unpack(tf.transpose(self.weights_sent_decwords_batch,perm=[1,0,2]))
      weights_words_decsent=tf.unpack(tf.transpose(self.weights_words_decsent_batch,perm=[1,0,2]))
      enc_lens = self.enc_input_lens
      sent_enc_lens = self.sent_enc_input_lens
      
      
      with tf.variable_scope('embedding'): 
        embedding = tf.get_variable(
            'word_embedding',dtype=tf.float32,
            initializer=self.embed)
        emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in encoder_inputs]
        emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in decoder_inputs]


      with tf.variable_scope('sent_embedding'):
        sent_embedding = tf.get_variable(
            'sent_embedding', [hps.sent_enc_timesteps, hps.emb_dim], dtype=tf.float32)
            
        sent_emb_decoder_inputs = [tf.nn.embedding_lookup(sent_embedding, x)
                              for x in sent_decoder_inputs]

      for layer_i in xrange(hps.enc_layers):
        with tf.variable_scope('encoder%d'%layer_i):
	  emb_encoder_inputs=tf.unpack(tf.nn.dropout(emb_encoder_inputs,0.5))
          cell_fw = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden/2,
              initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=123),
              state_is_tuple=False)
          cell_bw = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden/2,
              initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=123),
              state_is_tuple=False)
          (emb_encoder_inputs, fw_state, bw_state) = tf.nn.bidirectional_rnn(
              cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32,
              sequence_length=enc_lens)


          
          
          
          
          
          
          
          
      encoder_outputs = emb_encoder_inputs

      
      sent_i=tf.transpose(encoder_outputs,perm=[1,0,2])
      
      index=tf.transpose(sent_encoder_inputs,perm=[1,0])
      
      sent_ip=tf.pack([tf.gather(sent_i[l],index[l]) for l in xrange(hps.batch_size)])
      sent_input=tf.unpack(tf.transpose(sent_ip,perm=[1,0,2]))





      for layer_i in xrange(hps.enc_layers):
        with tf.variable_scope('sent_encoder%d'%layer_i):
	  sent_input=tf.unpack(tf.nn.dropout(sent_input,0.5))
          cell_sent = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
              state_is_tuple=False)
          
          (sent_input, sent_fw_state) = tf.nn.rnn(
              cell_sent, sent_input, dtype=tf.float32,
              sequence_length=sent_enc_lens)
	  
      sent_encoder_outputs = sent_input
      


      with tf.variable_scope('decoder'):
        
        
        loop_function = None
        sent_loop_function = None
  

        self.cell = tf.nn.rnn_cell.LSTMCell(
            hps.num_hidden,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
            state_is_tuple=False)
	
        encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, hps.num_hidden])
                           for x in encoder_outputs]
        enc_top_states = tf.concat(1, encoder_outputs)
        
        dec_in_state=tf.concat(1,[fw_state,bw_state])




        with tf.variable_scope('sent_decoder'):
          self.sent_cell = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
              state_is_tuple=False)
    
        sent_encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, hps.num_hidden])
                           for x in sent_encoder_outputs]
        sent_enc_top_states = tf.concat(1, sent_encoder_outputs)

        
        if hps.mode== 'train':
          mode=True
        else:
          mode=False

        sent_dec_in_state = sent_fw_state
        sent_initial_state_attention = True 
	self.decoder_outputs, self.dec_out_state,self.sent_decoder_outputs, self.sent_dec_out_state,self.switch_output,self.switch_prob,self.decoder_outputs_dists,self.sent_decoder_outputs_dists = seq2seq.attention_decoder(
            emb_decoder_inputs,encoder_inputs, dec_in_state, enc_top_states,self.cell,
            sent_emb_decoder_inputs, sent_input,sent_dec_in_state, sent_enc_top_states,
            self.sent_cell,hps.dec_timesteps,switch=switch,word_weights=word_weights, mode_train=mode,num_heads=1, loop_function=loop_function,sent_loop_function=sent_loop_function,
            initial_state_attention=sent_initial_state_attention)

        switch_target=[tf.to_int32(tf.greater_equal(x,1)) for x in switch]



        final_dists = self._calc_final_dist(self.decoder_outputs_dists, self.sent_decoder_outputs_dists)
        

        log_dists = [tf.log(dist+1e-12) for dist in final_dists]
        with tf.variable_scope('loss'):
 
          loss_per_step = [] 
          batch_nums = tf.range(0, limit=hps.batch_size) 
	  sent_lens=1
	  word_lens=1
          for dec_step, log_dist in enumerate(log_dists):
            target = extend_targets[dec_step] 
            indices = tf.stack( (batch_nums, target), axis=1) 
            losses = tf.gather_nd(-log_dist, indices) 
	    w=(word_weights[dec_step]/word_lens)+(switch[dec_step]/sent_lens)
            loss_per_step.append(losses*w)
         
        self.loss =tf.reduce_mean(sum(loss_per_step))
	self.final_log_dists=final_dists 

      if  hps.mode!='decode':
        with tf.variable_scope('word_loss'):
          self.word_loss=self.get_loss(
                self.decoder_outputs, targets,self.word_weights_batch)  
 
        with tf.variable_scope('sent_loss'):
          self.sent_loss=self.get_loss(
                self.sent_decoder_outputs_dists, sent_targets,self.switch_batch) 
          

	with tf.variable_scope('switch_loss'):
          self.switch_loss=seq2seq.sequence_loss(
                self.switch_output,switch_target, switch_weights,
                softmax_loss_function=None)
  
	self.total_loss=self.loss+self.word_loss+self.sent_loss
        tf.scalar_summary('loss',tf.minimum(12.0,  self.loss))
 
  def add_train_op(self):
    """Sets self.train_op, op to run for training."""
    hps = self.hps
    loss_ind=self.loss_ind
  
    self.loss_to_minimize=self.loss +self.switch_loss+self.word_loss

    self.lr_rate = tf.maximum(
        hps.min_lr,  
        tf.train.exponential_decay(hps.lr, self.global_step, 30000, 0.98))

    tvars = tf.trainable_variables()

    with tf.device('/gpu:1'):
       grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self.loss_to_minimize, tvars), hps.max_grad_norm)


    

    optimizer=tf.train.AdamOptimizer()


    with tf.device('/gpu:1'):
          self.train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')

  
  def build_graph(self,embed):
   
    self.embed=tf.convert_to_tensor(embed,name="embed",dtype=tf.float32)
    self.add_placeholders()
    with tf.device('/gpu:1'):
        self.add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.add_train_op()
    self.summaries = tf.merge_all_summaries()

