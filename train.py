import tensorflow as tf
import numpy as np
import readers 
from inference import inference
import time
from datetime import datetime
import os
import math
import config
import eval_util
import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

def train():
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		global_step = tf.Variable(0, trainable=False)
		video_id, labels, rgb, audio, num_frames = readers.input(True)
		coord = tf.train.Coordinator()

		lr = tf.train.exponential_decay(FLAGS.lr, global_step, FLAGS.decay_steps, FLAGS.learning_decay_rate, staircase=True)
		tf.summary.scalar('learning_rate', lr)
		opt = tf.train.AdamOptimizer(lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=1e-08, use_locking=False, name='Adam')
		grads = inference(rgb, audio, num_frames, label = labels, train = True)
		loss = tf.get_collection("loss")[0]
		predict = tf.get_collection("predict")[0]

		tvars = tf.trainable_variables()

		for var in tvars:
			tf.summary.histogram(var.op.name, var)

		for grad, var in grads:
			print var.op.name
			if grad is not None and type(grad) is not tf.IndexedSlices:
				tf.summary.histogram(var.op.name + '/gradients', grad)
			elif type(grad) is tf.IndexedSlices:
				print "This is a indexslice gradient"
				print grad.dense_shape
			else:
				print "There is a None gradient"

		apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
		variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

		train_op = tf.group(apply_gradient_op, variables_averages_op)

		saver = tf.train.Saver(tf.global_variables())
		summary_op = tf.summary.merge_all()
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
		config.intra_op_parallelism_threads = 10
		config.inter_op_parallelism_threads = 16
		sess = tf.Session(config=config)
		sess.run(init)
		tf.train.start_queue_runners(sess=sess, coord = coord)
		summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)


		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		#saver.restore(sess, ckpt.model_checkpoint_path)

		#loader.restore(sess, ckpt.model_checkpoint_path)

		for step in xrange(FLAGS.max_steps):
			
		
			start_time = time.time()
			_, loss_value, predict_value, labels_value, num_frames_value = sess.run([train_op, loss, predict, labels, num_frames])
			duration = time.time() - start_time

			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

			if step % 10 == 0:
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				hit_at_one = eval_util.calculate_hit_at_one(predict_value, labels_value)
				perr = eval_util.calculate_precision_at_equal_recall_rate(predict_value,labels_value)
				gap = eval_util.calculate_gap(predict_value, labels_value)

				format_str = ('%s: step %d, loss = %.2f, hit@one = %.2f, perr = %.2f, gap = %.2f, (%.1f examples/sec; %.3f ''sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, hit_at_one, perr, gap, examples_per_sec, sec_per_batch))

			if step % 100 == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)
				summary_writer.add_summary(utils.MakeSummary("Hit@1", hit_at_one), step)
				summary_writer.add_summary(utils.MakeSummary("Perr", perr), step)
				summary_writer.add_summary(utils.MakeSummary("Gap", gap), step)
				summary_writer.add_summary(utils.MakeSummary("example per second", examples_per_sec), step)

			if ( step % 1000 == 0 and step != 0 ) or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)




def main(argv):
	train()


if __name__ == '__main__':
	tf.app.run()

