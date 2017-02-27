from model import * 
import tensorflow as tf
import sys
from tensorflow.python.ops import tensor_array_ops, control_flow_ops, random_ops, ctc_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d

thismodule = sys.modules[__name__]

flags = tf.app.flags
FLAGS = flags.FLAGS

def inference(rgb, audio, num_frames, label = None, train = True):
	current_infer = getattr(thismodule, "inference{}".format(FLAGS.current_model))
	return current_infer(rgb, audio, num_frames, label = label, train = train)

def inference1(rgb, audio, num_frames, label = None, train = True):
	RNNCell = LSTMCell
	activate = tf.nn.elu

	with tf.device("/gpu:0"):
		with tf.variable_scope('embed_rgb') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.rgb_size, FLAGS.rnn_rgb_hidden], initializer = xavier_initializer())
			bias = variable_on_cpu('bias', shape = [FLAGS.rnn_rgb_hidden], initializer = tf.constant_initializer())


			rgb_inputs = activate(tf.matmul(tf.reshape(rgb, [-1, FLAGS.rgb_size]), embedding) + bias)
			rgb_inputs = tf.reshape(rgb_inputs, [FLAGS.batch_size, -1, FLAGS.rnn_rgb_hidden])
			rgb_inputs = dropout(rgb_inputs, train)

		with tf.variable_scope('embed_audio') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.audio_size, FLAGS.rnn_audio_hidden], initializer = xavier_initializer())
			bias = variable_on_cpu('bias', shape = [FLAGS.rnn_audio_hidden], initializer = tf.constant_initializer())

			audio_inputs = activate(tf.matmul(tf.reshape(audio, [-1, FLAGS.audio_size]), embedding) + bias)
			audio_inputs = tf.reshape(audio_inputs, [FLAGS.batch_size, -1, FLAGS.rnn_audio_hidden])
			audio_inputs = dropout(audio_inputs, train)

		with tf.variable_scope('rgb_birnn') as scope:

			cell_fw = RNNCell(FLAGS.rnn_rgb_hidden, FLAGS.rnn_rgb_hidden, scope = "rnn_cell_fw")
			cell_bw = RNNCell(FLAGS.rnn_rgb_hidden, FLAGS.rnn_rgb_hidden, scope = "rnn_cell_bw")

			output_fw, _, _ = RNN(cell_fw, rgb_inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(cell_bw, tf.reverse(rgb_inputs, [1]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [1])
			rgb_inputs = tf.concat([output_fw, output_bw], 2)
			rgb_inputs = dropout(rgb_inputs, train)

		with tf.variable_scope('audio_birnn') as scope:

			cell_fw = RNNCell(FLAGS.rnn_audio_hidden, FLAGS.rnn_audio_hidden, scope = "rnn_cell_fw")
			cell_bw = RNNCell(FLAGS.rnn_audio_hidden, FLAGS.rnn_audio_hidden, scope = "rnn_cell_bw")

			output_fw, _, _ = RNN(cell_fw, audio_inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(cell_bw, tf.reverse(audio_inputs, [1]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [1])
			audio_inputs = tf.concat([output_fw, output_bw], 2)

			audio_inputs = dropout(audio_inputs, train)

		with tf.variable_scope('rgb_rnns') as scope:

			rnn_cells = [RNNCell(FLAGS.rnn_rgb_hidden * 2, FLAGS.rnn_rgb_hidden * 2, scope = "rnn_cell_{}".format(x)) for x in xrange(0, FLAGS.rnn_rgb_layer)]
			current_input = rgb_inputs
			for x in xrange(0, FLAGS.rnn_rgb_layer):
				current_input, hidden, _ = RNN(rnn_cells[x], current_input, scope = "rnn_{}".format(x))
				current_input = dropout(current_input, train)
			rgb_inputs = dropout(hidden, train)

		with tf.variable_scope('audio_rnns') as scope:

			rnn_cells = [RNNCell(FLAGS.rnn_audio_hidden * 2, FLAGS.rnn_audio_hidden * 2, scope = "rnn_cell_{}".format(x)) for x in xrange(0, FLAGS.rnn_rgb_layer)]
			current_input = audio_inputs
			for x in xrange(0, FLAGS.rnn_rgb_layer):
				current_input, hidden, _ = RNN(rnn_cells[x], current_input, scope = "rnn_{}".format(x))
				current_input = dropout(current_input, train)
			audio_inputs = dropout(hidden, train)

		with tf.variable_scope('final') as scope:
			joint_inputs = tf.concat([rgb_inputs, audio_inputs], 1)
			joint_dim = (FLAGS.rnn_audio_hidden + FLAGS.rnn_rgb_hidden) * 2
			W1 = variable_on_cpu('W1', shape = [joint_dim, FLAGS.final_hidden], initializer = xavier_initializer())
			b1 = variable_on_cpu('b1', shape = [FLAGS.final_hidden], initializer = tf.constant_initializer())

			W2 = variable_on_cpu('W2', shape = [FLAGS.final_hidden, FLAGS.num_class], initializer = xavier_initializer())
			b2 = variable_on_cpu('b2', shape = [FLAGS.num_class], initializer = tf.constant_initializer())

			inputs = activate(tf.matmul(joint_inputs, W1) + b1)
			inputs = dropout(inputs, train)
			final_output = tf.sigmoid(tf.matmul(inputs, W2) + b2)
			tf.add_to_collection('predict', final_output)

		if not train:
			return final_output

		with tf.variable_scope('loss') as scope:
			loss = cross_entropy(label, final_output, num_frames)

			final_loss = loss

			tf.add_to_collection("loss", final_loss)
			tf.summary.scalar("loss", final_loss)

			tvars = tf.trainable_variables()
			grads = tf.gradients(final_loss, tvars, colocate_gradients_with_ops = True)
			grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
			grads = zip(grads[0], tvars)

		return grads

def inference0(rgb, audio, num_frames, label = None, train = True):
	RNNCell = LSTMCell
	activate = tf.nn.elu

	with tf.device("/gpu:0"):
		with tf.variable_scope('embed_rgb') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.rgb_size, FLAGS.rnn_rgb_hidden], initializer = xavier_initializer())
			bias = variable_on_cpu('bias', shape = [FLAGS.rnn_rgb_hidden], initializer = tf.constant_initializer())


			rgb_inputs = activate(tf.matmul(tf.reshape(rgb, [-1, FLAGS.rgb_size]), embedding) + bias)
			rgb_inputs = tf.reshape(rgb_inputs, [FLAGS.batch_size, -1, FLAGS.rnn_rgb_hidden])

		with tf.variable_scope('embed_audio') as scope:
			embedding = variable_on_cpu('embedding', shape = [FLAGS.audio_size, FLAGS.rnn_audio_hidden], initializer = xavier_initializer())
			bias = variable_on_cpu('bias', shape = [FLAGS.rnn_audio_hidden], initializer = tf.constant_initializer())

			audio_inputs = activate(tf.matmul(tf.reshape(audio, [-1, FLAGS.audio_size]), embedding) + bias)
			audio_inputs = tf.reshape(audio_inputs, [FLAGS.batch_size, -1, FLAGS.rnn_audio_hidden])

		with tf.variable_scope('rgb_birnn') as scope:

			cell_fw = RNNCell(FLAGS.rnn_rgb_hidden, FLAGS.rnn_rgb_hidden, scope = "rnn_cell_fw")
			cell_bw = RNNCell(FLAGS.rnn_rgb_hidden, FLAGS.rnn_rgb_hidden, scope = "rnn_cell_bw")

			output_fw, _, _ = RNN(cell_fw, rgb_inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(cell_bw, tf.reverse(rgb_inputs, [1]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [1])
			rgb_inputs = tf.concat([output_fw, output_bw], 2)

		with tf.variable_scope('audio_birnn') as scope:

			cell_fw = RNNCell(FLAGS.rnn_audio_hidden, FLAGS.rnn_audio_hidden, scope = "rnn_cell_fw")
			cell_bw = RNNCell(FLAGS.rnn_audio_hidden, FLAGS.rnn_audio_hidden, scope = "rnn_cell_bw")

			output_fw, _, _ = RNN(cell_fw, audio_inputs, scope = "rnn_fw")
			output_bw_temp, _, _ = RNN(cell_bw, tf.reverse(audio_inputs, [1]), scope = "rnn_bw")

			output_bw = tf.reverse(output_bw_temp, [1])
			audio_inputs = tf.concat([output_fw, output_bw], 2)

		with tf.variable_scope('rgb_rnns') as scope:

			rnn_cells = [RNNCell(FLAGS.rnn_rgb_hidden * 2, FLAGS.rnn_rgb_hidden * 2, scope = "rnn_cell_{}".format(x)) for x in xrange(0, FLAGS.rnn_rgb_layer)]
			current_input = rgb_inputs
			for x in xrange(0, FLAGS.rnn_rgb_layer):
				current_input, hidden, _ = RNN(rnn_cells[x], current_input, scope = "rnn_{}".format(x))
			rgb_inputs = hidden

		with tf.variable_scope('audio_rnns') as scope:

			rnn_cells = [RNNCell(FLAGS.rnn_audio_hidden * 2, FLAGS.rnn_audio_hidden * 2, scope = "rnn_cell_{}".format(x)) for x in xrange(0, FLAGS.rnn_rgb_layer)]
			current_input = audio_inputs
			for x in xrange(0, FLAGS.rnn_rgb_layer):
				current_input, hidden, _ = RNN(rnn_cells[x], current_input, scope = "rnn_{}".format(x))
			audio_inputs = hidden

		with tf.variable_scope('final') as scope:
			joint_inputs = tf.concat([rgb_inputs, audio_inputs], 1)
			joint_dim = (FLAGS.rnn_audio_hidden + FLAGS.rnn_rgb_hidden) * 2
			W1 = variable_on_cpu('W1', shape = [joint_dim, FLAGS.final_hidden], initializer = xavier_initializer())
			b1 = variable_on_cpu('b1', shape = [FLAGS.final_hidden], initializer = tf.constant_initializer())

			W2 = variable_on_cpu('W2', shape = [FLAGS.final_hidden, FLAGS.num_class], initializer = xavier_initializer())
			b2 = variable_on_cpu('b2', shape = [FLAGS.num_class], initializer = tf.constant_initializer())

			inputs = activate(tf.matmul(joint_inputs, W1) + b1)
			final_output = tf.sigmoid(tf.matmul(inputs, W2) + b2)
			tf.add_to_collection('predict', final_output)

		if not train:
			return final_output

		with tf.variable_scope('loss') as scope:
			loss = cross_entropy(label, final_output, num_frames)

			final_loss = loss

			tf.add_to_collection("loss", final_loss)
			tf.summary.scalar("loss", final_loss)

			tvars = tf.trainable_variables()
			grads = tf.gradients(final_loss, tvars, colocate_gradients_with_ops = True)
			grads = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
			grads = zip(grads[0], tvars)

		return grads




	





	

