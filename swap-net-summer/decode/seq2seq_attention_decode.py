
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

from data import read_data
import util
from pyrouge import Rouge155

_DIGIT_RE = re.compile(br"\s\d+\s|\s\d+$")
SYB_RE=re.compile(b"([.,!?\"':;)(])|--")

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_path', 'dataset/finished_files/test.bin', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('train_path', 'dataset/', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('rouge_dir', 'rouge/ROUGE-1.5.5/', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')


tf.app.flags.DEFINE_string('mode', 'test', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')


tf.app.flags.DEFINE_string('log_root', 'log_dir', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'exp_1', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_string('model_dir',
                           'model_decode', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('system_dir',
                           'system_decode', 'Path expression to tf.Example.')


tf.app.flags.DEFINE_integer('hidden_dim', 200, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 100, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 800, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 50, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('thresh', 3, 'sentence threshold decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('max_article_sentences', 50,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('vocabulary_size', 150000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')

tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')




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

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
 
  if running_avg_loss == 0:  
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss

  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)

  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss

def get_rouge(model_dir=FLAGS.model_dir,system_dir=FLAGS.system_dir):

  rouge_args = '-e rouge/ROUGE/data -n 4 -m  -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -l 100'
  r=Rouge155()
  r.model_dir=model_dir
  r.system_dir=system_dir
  r.model_filename_pattern='model.
  r.system_filename_pattern='system.(\d+).txt'
  output = r.convert_and_evaluate(rouge_args=rouge_args)
  print("Rouge score:")
  print(output)




def switch_fscore(final_dists,out_decoder_outputs,out_sent_decoder_outputs,target_labels,re_vocab,next_enc_batch,articles,abstracts,sent_enc_batch,start=0):


        trueNegative = 0
        truePositive = 0
        falseNegative = 0
        falsePositive = 0
        count=start

        
        final_dist=np.transpose(final_dists,axes=(1, 0, 2))
        sent_output=np.transpose(out_sent_decoder_outputs,axes=(1, 0, 2))
        word_output=np.transpose(out_decoder_outputs,axes=(1, 0, 2))


        for f_output,f_sent,f_word,f_target, ee_input,article,abstract,sent_ee_input in zip(final_dist,sent_output,word_output,target_labels,next_enc_batch,articles,abstracts,sent_enc_batch):
                p=0
                f=0
                if np.count_nonzero(ee_input)==0:
                        continue
                pred_sent_ind=[]
                pred_word_ind=[]
                prob_sent_ind=[]
                prob_word_ind=[]
                gnd_sent_ind=[]
                gnd_word_ind=[]
                pred_out=[]
                gnd_out=[]

                ipred_sent_ind=[]
                ipred_word_ind=[]
                iprob_sent_ind=[]
                iprob_word_ind=[]

                for  ff_output,ff_sent,ff_word,ff_target in zip(f_output,f_sent,f_word, f_target):
                        max_ind=np.argmax(ff_output)
                        max_ind_sent=np.argmax(ff_output[FLAGS.max_enc_steps:])
                        max_ind_word=np.argmax(ff_output[:FLAGS.max_enc_steps])
                        prob_sent=ff_output[FLAGS.max_enc_steps:][max_ind_sent]
                        prob_word=ff_output[:FLAGS.max_enc_steps][max_ind_word]
                        if max_ind == read_data.EOS_ID:
                                break
                        pred_out.append(max_ind)
                        pred_sent_ind.append(max_ind_sent)
                        pred_word_ind.append(max_ind_word)
                        gnd_out.append(ff_target)
			prob_word_ind.append(prob_word)
			prob_sent_ind.append(prob_sent)
			
                        imax_ind_sent=np.argmax(ff_sent)
                        imax_ind_word=np.argmax(ff_word)
                        iprob_sent=ff_sent[imax_ind_sent]
                        iprob_word=ff_word[imax_ind_word]

                        ipred_sent_ind.append(imax_ind_sent)
                        ipred_word_ind.append(imax_ind_word)
                        iprob_word_ind.append(iprob_word)
                        iprob_sent_ind.append(iprob_sent)

			

                get_new_rouge_words(ee_input,abstract,iprob_sent_ind,ipred_sent_ind,article,count,ipred_word_ind,iprob_word_ind,sent_ee_input,re_vocab)
		if FLAGS.mode =='test':
			break


    

def get_new_rouge_words(ee_input,abstracts,sent_prob,pred_sent_ind,article,count,pred_word_ind,prob_word_ind,sent_ee_input,re_vocab):


                eos_prob=0.0
                if read_data.EOS_ID in pred_sent_ind:
			z=pred_sent_ind.index(read_data.EOS_ID)
                        eos_prob=sent_prob[z]
                        sent_prob=sent_prob[:z]
                        pred_sent_ind=pred_sent_ind[:z]

		prob_sent_ind=sent_prob
                word_text=[re_vocab[ee_input[w]] for i,w in enumerate(pred_word_ind)]
		word_dict={}
		word_ind_dict={}
		for w,ind,p in zip(word_text,pred_word_ind,prob_word_ind):
			if w > read_data.NUL_ID:
				if w not in word_dict.keys() :
					word_dict[w]=[p,ind]
				else:
					if p > word_dict[w][0]:
						word_dict[w]=[p,ind]
		
		final_prob=[]
		
		for val in word_dict.values():
			for v in xrange(1,len(val)):
				word_ind_dict[val[v]]=val[0]
		for i,s in enumerate(pred_sent_ind):
			words_out=[]
			if s > read_data.NUL_ID:
				words=range(sent_ee_input[s-1]+1,sent_ee_input[s]+1)

				words_out=[word_ind_dict[w] for w in words if w in word_ind_dict.keys()]
	
			if len(words_out)==0:
				words_out=[0]
			total_prob= prob_sent_ind[i]+np.sum(words_out)
			final_prob.append(total_prob)
	

                sent_prob_ind=np.array(final_prob).argsort()
                max_prob_ind=list(reversed(sent_prob_ind))
                max_prob=[pred_sent_ind[i] for i in max_prob_ind]
                max_sent_prob=[]
                for i in max_prob:
                        if (i not in max_sent_prob) and (i != read_data.EOS_ID):
                                max_sent_prob.append(i)
                if len(max_sent_prob) > FLAGS.thresh:
                        max_sent_prob=max_sent_prob[:FLAGS.thresh]
                out_ind=sorted(set(max_sent_prob))
		write_rouge(out_ind,article,abstracts,count)

def write_rouge(out_ind,article,abstracts,count):
                if out_ind != []:
                        
                        system_sents=[]
                        abstract_sents=[]
                        for output in out_ind:
                                if (output-5 >= 0 and output-5 < len(article)):
                                        system_text=' '.join([word for word in article[output-5]])
                                        
                                        system_text=_DIGIT_RE.sub(b" 0 ",system_text)
					system_text=' '.join([word for word in system_text.split()])
                                        system_sents.append(system_text)
                        model_file=open(os.path.join(FLAGS.model_dir, "model.%d.txt" % count),"w+")
                        system_file=open(os.path.join(FLAGS.system_dir, "system.%d.txt" % count),"w+")
                        for abstract in abstracts:
                                
                                abstract=_DIGIT_RE.sub(b" 0 ",abstract)
				abstract_text=' '.join([word for word in abstract.split()])
                                abstract_sents.append(abstract_text)
                        model_file.write("\n".join([tf.compat.as_str(sents) for sents in abstract_sents ]))
                        system_file.write("\n".join([tf.compat.as_str(sents) for sents in system_sents ]))



def run_decode(model, batcher, re_vocab,embed):
  model.build_graph(embed) 
  saver = tf.train.Saver(max_to_keep=3) 
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") 

  running_avg_loss = 0 
  best_loss = None  
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
    os.makedirs(FLAGS.system_dir)

  ckpt_state = tf.train.get_checkpoint_state(eval_dir,latest_filename="checkpoint_best")
  tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  print('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  saver.restore(sess, ckpt_state.model_checkpoint_path)
  count=0
  while True:

          batch = batcher.next_batch() 
          if batch is None:
                get_rouge()
  
          step_output= model.run_eval_step(sess,batch)


          ( summaries, loss, train_step)=step_output[0]
	  (out_decoder_outputs,out_sent_decoder_outputs,final_dists)=step_output[1]
          (step_loss,word_loss,sent_loss,word_loss_null,sent_loss_null,switch_loss)=step_output[2]

          switch_fscore(final_dists,out_decoder_outputs,out_sent_decoder_outputs,batch.extend_target_batch,re_vocab,batch.enc_batch,batch.origin_articles,batch.origin_abstracts,batch.sent_enc_batch,count)

          count=count+1
                                                                                                                     

def main(unused_argv):
  if len(unused_argv) != 1: 
    raise Exception("Problem with flags: %s" % unused_argv)

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
      num_softmax_samples=40)  


  
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
