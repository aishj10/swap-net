
import sys
import time


import tensorflow as tf
from tensorflow.python.platform import gfile
from data import batch_reader

from model import seq2seq_attention_model

import random
import os
import numpy as np
import re
import warnings

from data import read_data
import util
from pyrouge import Rouge155

_DIGIT_RE = re.compile(br"\s\d+\s|\s\d+$")
SYB_RE=re.compile(b"([.,!?\"':;)(])|--")


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_path', 'dataset/finished_files/val.bin', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('train_path', 'dataset/finished_files', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('rouge_dir', 'rouge/ROUGE-1.5.5/', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')


tf.app.flags.DEFINE_string('mode', 'eval', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')


tf.app.flags.DEFINE_string('log_root', 'log_dir', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'exp_1', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_string('model_dir',
                           'model_dir_eval', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('system_dir',
                           'system_dir_eval', 'Path expression to tf.Example.')


tf.app.flags.DEFINE_integer('hidden_dim', 200, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 100, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 800, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 50, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('thresh', 3, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('max_article_sentences', 50,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('vocabulary_size', 150000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')

tf.app.flags.DEFINE_float('max_grad_norm', 1.5, 'for gradient clipping')


thresh=3
print('Parameters:',tf.flags.FLAGS.__flags)
print('Notes: Copied from wn_2, switch is removed,added word and sent null losses,changed NUL_ID to 0 from 4')

def prepare_vocab():
    """Fill input queue with ModelInput."""
    vocab_path = os.path.join(FLAGS.train_path, "vocab%d.txt" % FLAGS.vocabulary_size)
    key_path = os.path.join(FLAGS.train_path, "keyword%d.txt" % FLAGS.vocabulary_size)
  
    if not (gfile.Exists(vocab_path) and gfile.Exists(key_path)):
      print('No Vocabulary and keywords exist')
      
    print('Reading Vocabulary and keywords')
    vocab, re_vocab = read_data.initialize_vocabulary(vocab_path)
    key=read_data.initialize_keywords(key_path)
    embed=read_data.get_embedding(vocab,FLAGS.emb_dim)
    return vocab,re_vocab,key,embed

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.55):

  if running_avg_loss == 0:  
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * (1-decay) + (decay) * loss

  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)

  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss




def run_eval(model, batcher,re_vocab,embed):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph(embed) 
  saver = tf.train.Saver(max_to_keep=10) 
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") 
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') 
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 
  best_loss = None  
  count=0


  while True:
    _ = util.load_ckpt(saver, sess) 
    batch = batcher.next_batch() 

    if not os.path.exists(FLAGS.model_dir):
      os.makedirs(FLAGS.model_dir)
      os.makedirs(FLAGS.system_dir)

    
    t0=time.time()

    step_output= model.run_eval_step(sess,batch)
    t1=time.time()

    
    (summaries, loss, train_step)=step_output[0]
    (out_decoder_outputs,out_sent_decoder_outputs,final_dists)=step_output[1]
    (step_loss,word_loss,sent_loss,word_loss_null,sent_loss_null,switch_loss)=step_output[2]
    coverage_loss=0.0 

  
    running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)
  
    if best_loss is None or running_avg_loss < (best_loss):
      if best_loss is None:
	best_loss=0.0
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
   
      best_loss = running_avg_loss
      last_step=train_step


     

    tf.logging.info('loss: %f rloss: %f', loss,rloss) 
    tf.logging.info('step_loss: %f word_loss: %f ,sent_loss: %f ,word_loss_null: %f,sent_loss_null: %f ,switch_loss: %f,cover_loss: %f',step_loss,word_loss,sent_loss,word_loss_null,sent_loss_null,switch_loss,coverage_loss)

    os.system("rm -rf " + FLAGS.model_dir + ' ' +FLAGS.system_dir)

    count=count+1

    if train_step % 100 == 0:
      summary_writer.flush()
                                                                                                                                                       


                                                                                                                       

def main_script():
  tf.logging.set_verbosity(tf.logging.INFO) 
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))
  print('Initialising')
  
  vocab,re_vocab,key,embed = prepare_vocab()
  voacb_start=[read_data.PAD_ID,read_data.GO_ID,read_data.EOS_ID,read_data.UNK_ID,read_data.NUL_ID]
  
  start_enc=len(voacb_start)
  
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

 
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
    os.makedirs(FLAGS.system_dir)
  

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  
      min_lr=0.01,  
      lr=FLAGS.lr ,
      batch_size=FLAGS.batch_size,
      enc_layers=1,
      enc_timesteps=FLAGS.max_enc_steps,
      dec_timesteps=FLAGS.max_dec_steps, 
      sent_enc_timesteps=FLAGS.max_article_sentences+start_enc,
      max_article_sentences=FLAGS.max_article_sentences,
      min_input_len=2,  
      num_hidden=FLAGS.hidden_dim,  
      emb_dim=FLAGS.emb_dim,  
      vocabulary_size=FLAGS.vocabulary_size,
      max_grad_norm=2,
      num_softmax_samples=0)  


  
  batcher = batch_reader.Batcher(
      FLAGS.data_path,vocab,hps,start_enc, single_pass=FLAGS.single_pass)

  tf.set_random_seed(111) 

  if hps.mode == 'train':
    print "creating model..."

    model = seq2seq_attention_model.Seq2SeqAttentionModel(
				hps, vocab, num_gpus=2)
    setup_training(model, batcher,re_vocab,embed)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=2)
    run_eval(model, batcher,re_vocab,embed)
  elif hps.mode == 'test':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=2)
    run_decode(model, batcher,re_vocab,embed)

  
  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
  tf.app.run()
