import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops, random_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d, batch_norm
import math
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def variable_on_cpu(name, shape=None, dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True):

	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape = shape, dtype = dtype, initializer = initializer, regularizer = regularizer, trainable = trainable, collections = collections, caching_device = caching_device, partitioner = partitioner, validate_shape = validate_shape)
	return var

def RNN(cell, inputs, initial_state = None, initial_hidden = None, parallel_iterations = 10, swap_memory = True, collections = None, scope = "rnn"):
	'''
	This is an dynamic implementation of RNN using while_loop and TensorArray.
	Inputs has shape [batch_size, time_step, feature_dim].
	Cell is a function of each RNN step.
	'''
	with tf.variable_scope(scope):

		#For restoring shape later
		shape = inputs.get_shape()
		batch_size_value = shape[0].value
		#change to time, batch, feature
		inputs = tf.transpose(inputs, [1, 0, 2])
		#get time_step tensor
		input_shape = tf.shape(inputs)
		time_step, batch_size, _ = tf.unpack(input_shape)

		#Create input and output tensorarray
		input_ta = tensor_array_ops.TensorArray(dtype = inputs.dtype, size = time_step, tensor_array_name = scope + "input")
		output_ta = tensor_array_ops.TensorArray(dtype = inputs.dtype, size = time_step, tensor_array_name = scope + "output")
		input_ta = input_ta.unpack(inputs)

		if initial_state is None:
			initial_state = tf.zeros([batch_size_value, cell.hidden_size])

		if initial_hidden is None:
			initial_hidden = tf.zeros([batch_size_value, cell.hidden_size])

		
		time = tf.constant(0, name = "time")
		
		def rnn_step(time, output_t, hidden_t, state_t):
			#get the input and write it to output
			input_t = input_ta.read(time)
			input_t.set_shape([shape[0], shape[2]])
			output, newstate = cell(input_t, hidden_t, state_t)
			output_t = output_t.write(time, output)

			return time + 1, output_t, output, newstate

		#Loop through sequence with rnn_step
		loop_vars = control_flow_ops.while_loop(
			cond = lambda time, *_: time < time_step,
			body = rnn_step,
			loop_vars = (time, output_ta, initial_hidden, initial_state),
			parallel_iterations = parallel_iterations,
			swap_memory = swap_memory
			)

		#Get the output and the last hidden output
		output_final_ta, final_hidden, final_state = loop_vars[1], loop_vars[2], loop_vars[3]

		#Pack to Tensor and change the axes back
		final_ouput = output_final_ta.pack()
		final_ouput = tf.transpose(final_ouput, [1, 0, 2])

		#Restore shape information
		final_ouput.set_shape([shape[0], shape[1], cell.hidden_size])

	return final_ouput, final_hidden, final_state

class LSTMCell(object):
	"""
	This is a quick implementation on LSTM Cell.
	There are following possible TODOs:
	1. Add Dropout support
	2. Add Batch Normalization support (On gate level or on sequence level)
	3. Combine variables together to optimize matmul for GPU.
	"""
	def __init__(self, hidden_size, input_size, extra_input = 0, input_dim_list = None, collections = None, init_forget = 1.0, scope = "LSTMCell"):
		super(LSTMCell, self).__init__()
		with tf.variable_scope(scope):
			self.hidden_size = hidden_size
			self.scope = scope
			self.input_size = input_size
			self.extra_input = extra_input

			#input gate
			self.W_i = variable_on_cpu("W_input", shape = [input_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.U_i = variable_on_cpu("U_input", shape = [hidden_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.b_i = variable_on_cpu("b_input", shape = [hidden_size], initializer = tf.constant_initializer(), collections = collections)

			#Forget gate
			self.W_f = variable_on_cpu("W_forget", shape = [input_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.U_f = variable_on_cpu("U_forget", shape = [hidden_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.b_f = variable_on_cpu("b_forget", shape = [hidden_size], initializer = tf.constant_initializer(init_forget), collections = collections)
			
			#Output gate
			self.W_o = variable_on_cpu("W_output", shape = [input_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.U_o = variable_on_cpu("U_output", shape = [hidden_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.b_o = variable_on_cpu("b_output", shape = [hidden_size], initializer = tf.constant_initializer(), collections = collections)

			#Candidate
			self.W = variable_on_cpu("W", shape = [input_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.U = variable_on_cpu("U", shape = [hidden_size, hidden_size], initializer = xavier_initializer(), collections = collections)
			self.b = variable_on_cpu("b", shape = [hidden_size], initializer = tf.constant_initializer(), collections = collections)

			self.i_list = []
			self.f_list = []
			self.o_list = []
			self.c_list = []

			
			self.collections = collections


			for x in xrange(0, extra_input):
				assert len(input_dim_list) == extra_input, "The length of input dim must match number of input"

				self.i_list.append(variable_on_cpu("extra_i_{}".format(x), shape = [input_dim_list[x], hidden_size], initializer = xavier_initializer(), collections = collections))
				self.f_list.append(variable_on_cpu("extra_f_{}".format(x), shape = [input_dim_list[x], hidden_size], initializer = xavier_initializer(), collections = collections))
				self.o_list.append(variable_on_cpu("extra_o_{}".format(x), shape = [input_dim_list[x], hidden_size], initializer = xavier_initializer(), collections = collections))
				self.c_list.append(variable_on_cpu("extra_c_{}".format(x), shape = [input_dim_list[x], hidden_size], initializer = xavier_initializer(), collections = collections))

	#Todo add dropout and batch normalization
	def __call__(self, inputs, hidden_state, cell_state, context_list = None):
		with tf.variable_scope(self.scope):

			if self.extra_input != 0:

				i = tf.sigmoid(tf.matmul(inputs , self.W_i) + tf.matmul(hidden_state , self.U_i)  + tf.add_n([tf.matmul(c, w)for c, w in zip(context_list, self.i_list)]) + self.b_i, name = "input_gate")
				f = tf.sigmoid(tf.matmul(inputs , self.W_f) + tf.matmul(hidden_state , self.U_f)  + tf.add_n([tf.matmul(c, w)for c, w in zip(context_list, self.f_list)]) + self.b_f, name = "forget_gate")
				o = tf.sigmoid(tf.matmul(inputs , self.W_o) + tf.matmul(hidden_state , self.U_o)  + tf.add_n([tf.matmul(c, w)for c, w in zip(context_list, self.o_list)]) + self.b_o, name = "output_gate")
				candidate = tf.tanh( tf.matmul(inputs , self.W) + tf.matmul(hidden_state, self.U) + tf.add_n([tf.matmul(c, w)for c, w in zip(context_list, self.c_list)]) + self.b, name = "candidate")
				
				new_cell_state = cell_state * f  + i * candidate 
				new_hidden_state = tf.mul(tf.tanh(new_cell_state), o, name = "hidden_state")
				
			else:
				i = tf.sigmoid(tf.matmul(inputs , self.W_i) + tf.matmul(hidden_state , self.U_i)  + self.b_i, name = "input_gate")
				f = tf.sigmoid(tf.matmul(inputs , self.W_f) + tf.matmul(hidden_state , self.U_f)  + self.b_f, name = "forget_gate")
				o = tf.sigmoid(tf.matmul(inputs , self.W_o) + tf.matmul(hidden_state , self.U_o)  + self.b_o, name = "output_gate")
				candidate = tf.tanh( tf.matmul(inputs , self.W) + tf.matmul(hidden_state, self.U) + self.b, name = "candidate")
				
				new_cell_state = cell_state * f  + i * candidate 
				new_hidden_state = tf.mul(tf.tanh(new_cell_state), o, name = "hidden_state")

		return new_hidden_state, new_cell_state

def dropout(inputs, istraining, keep_prob=0.9, noise_shape=None, scope=None):
	if istraining:
		outputs = tf.nn.dropout(inputs, keep_prob, noise_shape)
	else:
		outputs = tf.nn.dropout(inputs, 1.0, noise_shape)
	return outputs

def inference6(string, feature, train, label = None):
	#model 2
	with tf.device("/gpu:0"):
		with tf.variable_scope('embed') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.string_dim, FLAGS.rnn_hidden * 2], initializer = xavier_initializer())
			inputs = tf.nn.embedding_lookup(embedding, string)
		
			inputs = batch_norm(inputs, updates_collections=None, is_training=True)
			inputs = tf.nn.elu(inputs)

			

		with tf.variable_scope('rnn') as scope:
			lstm_fw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_fw") 
			lstm_bw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_bw")
			lstm_2 = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_2")
			lstm_3 = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden , scope = "lstm_3")

			output_fw, _, _ = RNN(lstm_fw, inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(lstm_bw, tf.reverse(inputs, [False, True, False]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [False, True, False])
			inputs = tf.concat(2, [output_fw, output_bw])

			inputs = batch_norm(inputs, updates_collections=None, is_training=True)
			full_output, final_hidden, _ = RNN(lstm_2, inputs, scope = "rnn_2")
			
			inputs = batch_norm(full_output, updates_collections=None,  is_training=True)
			full_output, final_hidden, _ = RNN(lstm_3, inputs, scope = "rnn_3")


		with tf.variable_scope('vanilla') as scope:
			W1 = variable_on_cpu('W1', shape = [FLAGS.feature_dim, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			W2 = variable_on_cpu('W2', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			
			inputs = tf.matmul(feature, W1)
			inputs = batch_norm(inputs,updates_collections=None,  is_training=True)
			inputs = tf.nn.elu(inputs)
			
			inputs = tf.matmul(inputs, W2)
			inputs = batch_norm(inputs,updates_collections=None,  is_training=True)
			vanilla_input = tf.nn.elu(inputs)


		with tf.variable_scope('final') as scope:
			joint_inputs = tf.concat(1, [final_hidden, vanilla_input])
			joint_dim = FLAGS.rnn_hidden + FLAGS.vanilla_hidden
			W1 = variable_on_cpu('W1', shape = [joint_dim, FLAGS.final_hidden], initializer = xavier_initializer())
			W2 = variable_on_cpu('W2', shape = [FLAGS.final_hidden, FLAGS.final_hidden], initializer = xavier_initializer())
			W3 = variable_on_cpu('W3', shape = [FLAGS.final_hidden, 1], initializer = xavier_initializer())

			
			inputs = tf.matmul(joint_inputs, W1)
			inputs = batch_norm(inputs, updates_collections=None, is_training=True)
			inputs = tf.nn.elu(inputs)

			inputs = tf.matmul(inputs, W2)
			inputs = batch_norm(inputs,updates_collections=None,  is_training=True)
			inputs = tf.nn.elu(inputs)

			inputs = batch_norm(inputs,updates_collections=None,  is_training=True)
			inputs = tf.nn.elu(inputs)

			final_ouput = tf.matmul(inputs, W3)
			final_ouput = tf.squeeze(final_ouput)

		if not train:
			return final_ouput

		with tf.variable_scope('loss') as scope:
			loss = tf.nn.l2_loss(label - final_ouput)
			tf.add_to_collection("loss", loss)
			tf.summary.scalar(loss.op.name, loss)


		tvars = tf.trainable_variables()
		grads = tf.gradients(loss, tvars, colocate_gradients_with_ops = True)
		grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
		grads = zip(grads[0], tvars)

		return grads

def inference5(string, feature, train, label = None):
	#model 1
	with tf.device("/gpu:0"):
		with tf.variable_scope('embed') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.string_dim, FLAGS.rnn_hidden * 2], initializer = xavier_initializer())
			bias = variable_on_cpu('bias', shape = [FLAGS.rnn_hidden * 2], initializer = tf.constant_initializer())

			inputs = tf.nn.embedding_lookup(embedding, string)
			inputs = tf.nn.elu(tf.nn.bias_add(inputs, bias))
			inputs = dropout(inputs, train)

		with tf.variable_scope('rnn') as scope:
			lstm_fw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_fw") 
			lstm_bw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_bw")
			lstm_2 = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_2")

			output_fw, _, _ = RNN(lstm_fw, inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(lstm_bw, tf.reverse(inputs, [False, True, False]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [False, True, False])
			inputs = tf.concat(2, [output_fw, output_bw])
		
			full_output, final_hidden, _ = RNN(lstm_2, inputs, scope = "rnn_2")
			final_hidden = dropout(final_hidden, train)

		with tf.variable_scope('vanilla') as scope:
			W1 = variable_on_cpu('W1', shape = [FLAGS.feature_dim, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			b1 = variable_on_cpu('bias1', shape = [FLAGS.vanilla_hidden], initializer = tf.constant_initializer())
			
			W2 = variable_on_cpu('W2', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			b2 = variable_on_cpu('bias2', shape = [FLAGS.vanilla_hidden], initializer = tf.constant_initializer())

			layer1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(feature, W1), b1))
			layer1 = dropout(layer1, train)
			layer2 = tf.nn.elu(tf.nn.bias_add(tf.matmul(layer1, W2), b2))
			layer2 = dropout(layer2, train)

		with tf.variable_scope('final') as scope:
			joint_inputs = tf.concat(1, [final_hidden, layer2])
			joint_dim = FLAGS.rnn_hidden + FLAGS.vanilla_hidden
			W1 = variable_on_cpu('W1', shape = [joint_dim, FLAGS.final_hidden], initializer = xavier_initializer())
			b1 = variable_on_cpu('bias1', shape = [FLAGS.final_hidden], initializer = tf.constant_initializer())
			
			W2 = variable_on_cpu('W2', shape = [FLAGS.final_hidden, 1], initializer = xavier_initializer())
			b2 = variable_on_cpu('bias2', shape = [1], initializer = tf.constant_initializer())

			layer1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(joint_inputs, W1), b1))
			layer1 = dropout(layer1, train)
			final_ouput = tf.nn.bias_add(tf.matmul(layer1, W2), b2)
			final_ouput = tf.squeeze(final_ouput)

		if not train:
			return final_ouput

		with tf.variable_scope('loss') as scope:
			loss = tf.nn.l2_loss(label - final_ouput)
			tf.add_to_collection("loss", loss)
			tf.summary.scalar(loss.op.name, loss)


		tvars = tf.trainable_variables()
		grads = tf.gradients(loss, tvars, colocate_gradients_with_ops = True)
		grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
		grads = zip(grads[0], tvars)

		return grads

def inference4(string, feature, train, label = None):
	#model 2
	with tf.device("/gpu:0"):
		

		with tf.variable_scope('vanilla') as scope:
			W1 = variable_on_cpu('W1', shape = [FLAGS.feature_dim, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			W2 = variable_on_cpu('W2', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			W3 = variable_on_cpu('W3', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			W4 = variable_on_cpu('W4', shape = [FLAGS.vanilla_hidden, 1], initializer = xavier_initializer())
			b = variable_on_cpu("b", shape = [1], initializer = tf.constant_initializer())


			inputs = batch_norm(feature, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer1 = tf.matmul(inputs, W1)
			residual_inputs = layer1

			inputs = batch_norm(layer1, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer2 = tf.matmul(inputs, W2)
			inputs = residual_inputs + layer2
			residual_inputs = inputs

			inputs = batch_norm(inputs, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer3 = tf.matmul(inputs, W3)
			vanilla_input = residual_inputs + layer3

			final_ouput = tf.nn.bias_add(tf.matmul(vanilla_input, W4), b)
			final_ouput = tf.squeeze(final_ouput)


		if not train:
			return final_ouput

		with tf.variable_scope('loss') as scope:
			loss = tf.nn.l2_loss(label - final_ouput)
			tf.add_to_collection("loss", loss)
			tf.summary.scalar(loss.op.name, loss)


		tvars = tf.trainable_variables()
		grads = tf.gradients(loss, tvars, colocate_gradients_with_ops = True)
		grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
		grads = zip(grads[0], tvars)

		return grads

def inference3(string, feature, train, label = None):
	#model 2
	with tf.device("/gpu:0"):
		with tf.variable_scope('embed') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.string_dim, FLAGS.rnn_hidden * 2], initializer = xavier_initializer())
			inputs = tf.nn.embedding_lookup(embedding, string)
		
			inputs = batch_norm(inputs,  updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)


		with tf.variable_scope('rnn') as scope:
			lstm_fw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_fw") 
			lstm_bw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_bw")
			lstm_2 = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_2")

			output_fw, _, _ = RNN(lstm_fw, inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(lstm_bw, tf.reverse(inputs, [False, True, False]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [False, True, False])
			inputs = tf.concat(2, [output_fw, output_bw])

			inputs = batch_norm(inputs, updates_collections=None, is_training=train)
			full_output, final_hidden, _ = RNN(lstm_2, inputs, scope = "rnn_2")
			

		with tf.variable_scope('vanilla') as scope:
			W1 = variable_on_cpu('W1', shape = [FLAGS.feature_dim, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			W2 = variable_on_cpu('W2', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			W3 = variable_on_cpu('W3', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())


			inputs = batch_norm(feature, updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer1 = tf.matmul(inputs, W1)
			residual_inputs = layer1

			inputs = batch_norm(layer1, updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer2 = tf.matmul(inputs, W2)
			inputs = residual_inputs + layer2
			residual_inputs = inputs

			inputs = batch_norm(inputs, updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer3 = tf.matmul(inputs, W3)
			vanilla_input = residual_inputs + layer3


		with tf.variable_scope('final') as scope:
			joint_inputs = tf.concat(1, [final_hidden, vanilla_input])
			joint_dim = FLAGS.rnn_hidden + FLAGS.vanilla_hidden
			W1 = variable_on_cpu('W1', shape = [joint_dim, FLAGS.final_hidden], initializer = xavier_initializer())
			W2 = variable_on_cpu('W2', shape = [FLAGS.final_hidden, FLAGS.final_hidden], initializer = xavier_initializer())
			W3 = variable_on_cpu('W3', shape = [FLAGS.final_hidden, 1], initializer = xavier_initializer())
			b = variable_on_cpu('b', shape = [1], initializer = tf.constant_initializer())

			inputs = batch_norm(joint_inputs, updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer1 = tf.matmul(inputs, W1)
			residual_inputs = layer1

			inputs = batch_norm(layer1, updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer2 = tf.matmul(inputs, W2)
			layer2 = residual_inputs + layer2
			
			final_ouput = tf.nn.bias_add(tf.matmul(layer2, W3), b)
			final_ouput = tf.squeeze(final_ouput)

		if not train:
			return final_ouput

		with tf.variable_scope('loss') as scope:
			loss = tf.nn.l2_loss(label - final_ouput)
			tf.add_to_collection("loss", loss)
			tf.summary.scalar(loss.op.name, loss)


		tvars = tf.trainable_variables()
		grads = tf.gradients(loss, tvars, colocate_gradients_with_ops = True)
		grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
		grads = zip(grads[0], tvars)

		return grads

def inference2(string, feature, train, label = None):
	#model 2
	with tf.device("/gpu:0"):
		with tf.variable_scope('embed') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.string_dim, FLAGS.rnn_hidden * 2], initializer = xavier_initializer())
			inputs = tf.nn.embedding_lookup(embedding, string)
		
			inputs = batch_norm(inputs, updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)

			

		with tf.variable_scope('rnn') as scope:
			lstm_fw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_fw") 
			lstm_bw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_bw")
			lstm_2 = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_2")
			lstm_3 = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden , scope = "lstm_3")

			output_fw, _, _ = RNN(lstm_fw, inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(lstm_bw, tf.reverse(inputs, [False, True, False]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [False, True, False])
			inputs = tf.concat(2, [output_fw, output_bw])

			inputs = batch_norm(inputs, updates_collections=None, is_training=train)
			full_output, final_hidden, _ = RNN(lstm_2, inputs, scope = "rnn_2")
			residual_inputs = final_hidden

			inputs = batch_norm(full_output,updates_collections=None,  is_training=train)
			full_output, final_hidden, _ = RNN(lstm_3, inputs, scope = "rnn_3")
			final_hidden = final_hidden + residual_inputs


		with tf.variable_scope('vanilla') as scope:
			W1 = variable_on_cpu('W1', shape = [FLAGS.feature_dim, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			W2 = variable_on_cpu('W2', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())

			W3 = variable_on_cpu('W3', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())


			inputs = batch_norm(feature,updates_collections=None,  is_training=train)
			inputs = tf.nn.elu(inputs)
			layer1 = tf.matmul(inputs, W1)
			residual_inputs = layer1

			inputs = batch_norm(layer1,updates_collections=None,  is_training=train)
			inputs = tf.nn.elu(inputs)
			layer2 = tf.matmul(inputs, W2)
			inputs = residual_inputs + layer2
			residual_inputs = inputs

			inputs = batch_norm(inputs,updates_collections=None,  is_training=train)
			inputs = tf.nn.elu(inputs)
			layer3 = tf.matmul(inputs, W3)
			vanilla_input = residual_inputs + layer3


		with tf.variable_scope('final') as scope:
			joint_inputs = tf.concat(1, [final_hidden, vanilla_input])
			joint_dim = FLAGS.rnn_hidden + FLAGS.vanilla_hidden
			W1 = variable_on_cpu('W1', shape = [joint_dim, FLAGS.final_hidden], initializer = xavier_initializer())
			W2 = variable_on_cpu('W2', shape = [FLAGS.final_hidden, FLAGS.final_hidden], initializer = xavier_initializer())
			W3 = variable_on_cpu('W3', shape = [FLAGS.final_hidden, FLAGS.final_hidden], initializer = xavier_initializer())
			W4 = variable_on_cpu('W4', shape = [FLAGS.final_hidden, 1], initializer = xavier_initializer())

			inputs = batch_norm(joint_inputs, updates_collections=None, is_training=train)
			inputs = tf.nn.elu(inputs)
			layer1 = tf.matmul(inputs, W1)
			residual_inputs = layer1

			inputs = batch_norm(layer1,updates_collections=None,  is_training=train)
			inputs = tf.nn.elu(inputs)
			layer2 = tf.matmul(inputs, W2)

			inputs = batch_norm(layer2,updates_collections=None,  is_training=train)
			inputs = tf.nn.elu(inputs)
			layer3 = tf.matmul(inputs, W3)

			inputs = residual_inputs + layer3
			final_ouput = tf.matmul(inputs, W4)
			final_ouput = tf.squeeze(final_ouput)

		if not train:
			return final_ouput

		with tf.variable_scope('loss') as scope:
			loss = tf.nn.l2_loss(label - final_ouput)
			tf.add_to_collection("loss", loss)
			tf.summary.scalar(loss.op.name, loss)


		tvars = tf.trainable_variables()
		grads = tf.gradients(loss, tvars, colocate_gradients_with_ops = True)
		grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
		grads = zip(grads[0], tvars)

		return grads



def inference1(string, feature, train, label = None):
	#model 1
	with tf.device("/gpu:0"):
		with tf.variable_scope('embed') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.string_dim, FLAGS.rnn_hidden * 2], initializer = xavier_initializer())
			bias = variable_on_cpu('bias', shape = [FLAGS.rnn_hidden * 2], initializer = tf.constant_initializer())

			inputs = tf.nn.embedding_lookup(embedding, string)
			inputs = tf.nn.elu(tf.nn.bias_add(inputs, bias))

		with tf.variable_scope('rnn') as scope:
			lstm_fw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_fw") 
			lstm_bw = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_bw")
			lstm_2 = LSTMCell(FLAGS.rnn_hidden, FLAGS.rnn_hidden * 2, scope = "lstm_2")

			output_fw, _, _ = RNN(lstm_fw, inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(lstm_bw, tf.reverse(inputs, [False, True, False]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [False, True, False])
			inputs = tf.concat(2, [output_fw, output_bw])
		
			full_output, final_hidden, _ = RNN(lstm_2, inputs, scope = "rnn_2")

		with tf.variable_scope('vanilla') as scope:
			W1 = variable_on_cpu('W1', shape = [FLAGS.feature_dim, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			b1 = variable_on_cpu('bias1', shape = [FLAGS.vanilla_hidden], initializer = tf.constant_initializer())
			
			W2 = variable_on_cpu('W2', shape = [FLAGS.vanilla_hidden, FLAGS.vanilla_hidden], initializer = xavier_initializer())
			b2 = variable_on_cpu('bias2', shape = [FLAGS.vanilla_hidden], initializer = tf.constant_initializer())

			layer1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(feature, W1), b1))
			layer2 = tf.nn.elu(tf.nn.bias_add(tf.matmul(layer1, W2), b2))

		with tf.variable_scope('final') as scope:
			joint_inputs = tf.concat(1, [final_hidden, layer2])
			joint_dim = FLAGS.rnn_hidden + FLAGS.vanilla_hidden
			W1 = variable_on_cpu('W1', shape = [joint_dim, FLAGS.final_hidden], initializer = xavier_initializer())
			b1 = variable_on_cpu('bias1', shape = [FLAGS.final_hidden], initializer = tf.constant_initializer())
			
			W2 = variable_on_cpu('W2', shape = [FLAGS.final_hidden, 1], initializer = xavier_initializer())
			b2 = variable_on_cpu('bias2', shape = [1], initializer = tf.constant_initializer())

			layer1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(joint_inputs, W1), b1))
			final_ouput = tf.nn.bias_add(tf.matmul(layer1, W2), b2)
			final_ouput = tf.squeeze(final_ouput)

		if not train:
			return final_ouput

		with tf.variable_scope('loss') as scope:
			loss = tf.nn.l2_loss(label - final_ouput)
			tf.add_to_collection("loss", loss)
			tf.summary.scalar(loss.op.name, loss)


		tvars = tf.trainable_variables()
		grads = tf.gradients(loss, tvars, colocate_gradients_with_ops = True)
		grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
		grads = zip(grads[0], tvars)

		return grads


def inference(string, feature, train, label = None):
	return inference4(string, feature, train, label = label)














