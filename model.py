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
		time_step, batch_size, _ = tf.unstack(input_shape)

		#Create input and output tensorarray
		input_ta = tensor_array_ops.TensorArray(dtype = inputs.dtype, size = time_step, tensor_array_name = scope + "input")
		output_ta = tensor_array_ops.TensorArray(dtype = inputs.dtype, size = time_step, tensor_array_name = scope + "output")
		input_ta = input_ta.unstack(inputs)

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
		final_ouput = output_final_ta.stack()
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
				new_hidden_state = tf.tanh(new_cell_state) * o
				
			else:
				i = tf.sigmoid(tf.matmul(inputs , self.W_i) + tf.matmul(hidden_state , self.U_i)  + self.b_i, name = "input_gate")
				f = tf.sigmoid(tf.matmul(inputs , self.W_f) + tf.matmul(hidden_state , self.U_f)  + self.b_f, name = "forget_gate")
				o = tf.sigmoid(tf.matmul(inputs , self.W_o) + tf.matmul(hidden_state , self.U_o)  + self.b_o, name = "output_gate")
				candidate = tf.tanh( tf.matmul(inputs , self.W) + tf.matmul(hidden_state, self.U) + self.b, name = "candidate")
				
				new_cell_state = cell_state * f  + i * candidate 
				new_hidden_state = tf.tanh(new_cell_state) * o

		return new_hidden_state, new_cell_state

def dropout(inputs, istraining, keep_prob=0.9, noise_shape=None, scope=None):
	if istraining:
		outputs = tf.nn.dropout(inputs, keep_prob, noise_shape)
	else:
		outputs = tf.nn.dropout(inputs, 1.0, noise_shape)
	return outputs

def cross_entropy(label, predict, num_frames, epsilon = 1e-6):
	float_label = tf.cast(label, tf.float32)
	cross_entropy_loss = float_label * tf.log(predict + epsilon) + (1 - float_label) * tf.log(1-predict+epsilon)
	cross_entropy_loss = tf.negative(cross_entropy_loss)
	loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))
	return loss















