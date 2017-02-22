import tensorflow as tf
import numpy as np
import readers 
from inference import inference
import time
from datetime import datetime
import os
import math
import config

flags = tf.app.flags
FLAGS = flags.FLAGS

def train():
	with tf.Graph().as_default(), tf.device('/cpu:0'):

		video_id, labels, rgb, audio, num_frames = readers.input(True)

		inference(rgb, audio, num_frames, labels)
		coord = tf.train.Coordinator()
		
		init = tf.global_variables_initializer()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		sess.run(init)

		#ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		#saver.restore(sess, ckpt.model_checkpoint_path)

		#summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

		tf.train.start_queue_runners(sess=sess, coord = coord)
		
		for step in xrange(FLAGS.max_steps):
			video_id_value, rgb_value, audio_value, labels_value, num_frames_value = sess.run([video_id, rgb, audio, labels, num_frames])


			print "video_id", video_id_value
			print "rgb", rgb_value
			print "rgb shape", rgb_value.shape
			print "audio", audio_value
			print "audio shape", audio_value.shape
			print "label", labels_value
			print "frame", num_frames_value









def main(argv):
	train()


if __name__ == '__main__':
	tf.app.run()

