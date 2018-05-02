

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
#from tensorflow.python.ops import rnn_cell_impl as rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest



# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding,batch, output_projection=None,
                              update_embedding=True):
 
  def loop_function(prev,encoder_inputs):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    # print("encoder_inputs",encoder_inputs)
    # print("math_ops.argmax(prev, 1)", math_ops.argmax(prev, 1))
    prev_symbol=[]
    ind = math_ops.argmax(prev, 1)
    # print(ind)
    prev_symbol=[]
    r=array_ops.transpose(encoder_inputs)
    for i in xrange(batch):
      ine=math_ops.to_int32(ind[i])
      prev_symbol.append(r[i,ine])
 
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function

def sent_extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):


  def sent_loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return sent_loop_function





def attention_decoder(decoder_inputs,
                      encoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      sent_decoder_inputs,
                      sent_encoder_inputs,
                      sent_initial_state,
                      sent_attention_states,
                      sent_cell,
                      dec_timesteps,
                      mode_train=True,
                      switch=None,
                      word_weights=None,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      sent_loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  
 
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype
    with variable_scope.variable_scope("word_attn") as attn_scope:

      batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
      attn_length = attention_states.get_shape()[1].value
      if attn_length is None:
        attn_length = shape(attention_states)[1]
      attn_size = attention_states.get_shape()[2].value

      # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
      hidden = array_ops.reshape(
          attention_states, [-1, attn_length, 1, attn_size])
      hidden_features = []
      v = []
      attention_vec_size = attn_size  # Size of query vectors for attention.
      for a in xrange(num_heads):
        k = variable_scope.get_variable("AttnW_%d" % a,
                                        [1, 1, attn_size, attention_vec_size])
        hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
        v.append(
            variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

   
    
      def attention(query,coverage=None):
        """Put attention masks on hidden using hidden_features and query."""
        ds = []  # Results of attention reads will be stored here.
        if nest.is_sequence(query):  # If the query is a tuple, flatten it.
          query_list = nest.flatten(query)
          for q in query_list:  # Check that ndims == 2 if specified.
            ndims = q.get_shape().ndims
            if ndims:
              assert ndims == 2
          query = array_ops.concat(1, query_list)
        for a in xrange(num_heads):
          with variable_scope.variable_scope("Attention_%d" % a):
            y = linear(query, attention_vec_size, True)
            y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
	
            s = math_ops.reduce_sum(
                  v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])

            atn = nn_ops.softmax(s)
  
            d = math_ops.reduce_sum(
                array_ops.reshape(atn, [-1, attn_length, 1, 1]) * hidden,
                [1, 2])
            ds.append(array_ops.reshape(d, [-1, attn_size]))
        return ds,s,atn 

      outputs = []
      #prev = None
      batch_attn_size = array_ops.pack([batch_size, attn_size])
      attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
               for _ in xrange(num_heads)]

      for a in attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, attn_size])
      if initial_state_attention:
        attns,ss,soft_ss = attention(initial_state)

    with variable_scope.variable_scope("sent_attn") as sent_attn_scope:

      sent_attn_length = sent_attention_states.get_shape()[1].value
      if sent_attn_length is None:
        sent_attn_length = shape(sent_attention_states)[1]
      sent_attn_size = sent_attention_states.get_shape()[2].value


      sent_hidden = array_ops.reshape(
          sent_attention_states, [-1, sent_attn_length, 1, sent_attn_size])
      sent_hidden_features = []
      sent_v = []
      sent_attention_vec_size = sent_attn_size  
      for a in xrange(num_heads):
        sent_k = variable_scope.get_variable("sent_AttnW_%d" % a,
                                        [1, 1, sent_attn_size, sent_attention_vec_size])
        sent_hidden_features.append(nn_ops.conv2d(sent_hidden, sent_k, [1, 1, 1, 1], "SAME"))
        sent_v.append(
            variable_scope.get_variable("sent_AttnV_%d" % a, [sent_attention_vec_size]))

    

      def sent_attention(query, sent_coverage=None):
        
        ds = []  # Results of attention reads will be stored here.
        if nest.is_sequence(query):  # If the query is a tuple, flatten it.
          query_list = nest.flatten(query)
          for q in query_list:  # Check that ndims == 2 if specified.
            ndims = q.get_shape().ndims
            if ndims:
              assert ndims == 2
          query = array_ops.concat(1, query_list)
        for a in xrange(num_heads):
          with variable_scope.variable_scope("sent_Attention_%d" % a):
            y = linear(query, sent_attention_vec_size, True)
            y = array_ops.reshape(y, [-1, 1, 1, sent_attention_vec_size])
  
            s = math_ops.reduce_sum(
                  v[a] * math_ops.tanh(sent_hidden_features[a] + y), [2, 3])

            atn = nn_ops.softmax(s)
      
            d = math_ops.reduce_sum(
                array_ops.reshape(atn, [-1, sent_attn_length, 1, 1]) * sent_hidden,
                [1, 2])
            ds.append(array_ops.reshape(d, [-1, sent_attn_size]))
        return ds,s,atn  

      outputs = []
      sent_outputs = []
      soft_outputs = []
      soft_sent_outputs = []
      
      sent_batch_attn_size = array_ops.pack([batch_size, sent_attn_size])
      sent_attns = [array_ops.zeros(sent_batch_attn_size, dtype=dtype)
               for _ in xrange(num_heads)]

      for a in sent_attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, sent_attn_size])
      if initial_state_attention:
        sent_attns,sent_ss,soft_sent_ss = sent_attention(sent_initial_state)


 
    hidden_words=[]
    hidden_sents=[]

    s_w=[]
    d_k = variable_scope.get_variable("switch_w",
                                    [1, 1, attn_size, attention_vec_size])
    T_k = variable_scope.get_variable("switch_s" ,
                                    [1, 1, sent_attn_size, sent_attention_vec_size])
    hidden_words=nn_ops.conv2d(hidden, d_k, [1, 1, 1, 1], "SAME")
    hidden_sents=nn_ops.conv2d(sent_hidden, T_k, [1, 1, 1, 1], "SAME")
 

    def switch_pos(st_w,st_s,h_w,h_s):
      with variable_scope.variable_scope("switch_w"):

        y_w=linear(st_w,2, True)

      with variable_scope.variable_scope("switch_s"):
        y_s=linear(st_s,2, True)
      with variable_scope.variable_scope("switch_hw"):
        y_hw=linear(h_w,2, True)
      with variable_scope.variable_scope("switch_hs"):
        y_hs=linear(h_s,2, True)

      s_b=y_s+y_hs+math_ops.tanh(y_w+y_hw)
      s_b=array_ops.reshape(s_b,[-1,2])
      switch_pb=nn_ops.softmax (s_b)
      return s_b ,switch_pb

    sent_state = sent_initial_state
    state = initial_state
    
    
    sent_prev = None
    prev=None


    
    switch_outputs=[]
    switch_softmax=[]
   
    for i in xrange(dec_timesteps):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
 
      
      sb,switch_prob=switch_pos(state,sent_state,attns,sent_attns)
      switch_outputs.append(sb)
      switch_softmax.append(switch_prob)
   
      inp=decoder_inputs[i]
      
      sent_inp=sent_decoder_inputs[i]
      if mode_train is not True and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          #print("Caliing Loop function")
	  if not loop_function ==None:
          	inp = loop_function(prev,encoder_inputs)
          	sent_inp=sent_loop_function(sent_prev,sent_encoder_inputs)
 
      input_size = inp.get_shape().with_rank(2)[1]
      sent_input_size = sent_inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)


      sent_switch=(switch_prob[:,1])
      word_switch=switch_prob[:,0] #ss(j-1)
 
 
      with variable_scope.variable_scope("word_stpes"):
        x=linear(array_ops.concat(2,[[inp],attns,math_ops.tanh(sent_attns)])[0],input_size,True)
        cell_output,state=cell(x,state)
	


      ##########Sentence decoder##################################


      with variable_scope.variable_scope("sent_steps"):

        sent_x=linear(array_ops.concat(2,[[sent_inp],sent_attns,math_ops.tanh(attns)])[0],sent_input_size,True)
        
        sent_cell_output,sent_state=sent_cell(sent_x,sent_state)
 
      with variable_scope.variable_scope(sent_attn_scope,reuse=True):
        sent_attns,sent_ss ,soft_sent_ss= sent_attention(sent_state)
      with variable_scope.variable_scope(attn_scope,reuse=True):
        attns,ss,soft_ss = attention(state)





      soft_ssout=soft_ss *array_ops.reshape([word_switch],[-1,1])
      soft_sent_ssout=soft_sent_ss *array_ops.reshape([sent_switch],[-1,1])

      prev=ss
      sent_prev=sent_ss
        
   
      outputs.append(soft_ss)
      soft_outputs.append(soft_ssout)

     
      sent_outputs.append(soft_sent_ss)
      soft_sent_outputs.append(soft_sent_ssout)

  
  return outputs, state, sent_outputs, sent_state,switch_outputs,switch_softmax,soft_outputs,soft_sent_outputs #,coverage




def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,switch_loss=False, name=None):
  
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets ):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
 
        if not switch_loss:
          target = array_ops.reshape(target, [-1])
          crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
              logit, target)
        else:
          target = array_ops.reshape(target, [-1])
          
          crossent = nn.sigmoid_cross_entropy_with_logits(
              logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None,switch_loss=False, name=None):
  
  with ops.name_scope(name, "sequence_loss", logits + targets ):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function,switch_loss=switch_loss))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


