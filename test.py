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
import csv

flags = tf.app.flags
FLAGS = flags.FLAGS


with tf.Graph().as_default(), tf.device('/cpu:0'):
	global_step = tf.Variable(0, trainable=False)
	k = 20
	video_id, labels, rgb, audio, num_frames = readers.input(False)
	coord = tf.train.Coordinator()
	
	predict = inference(rgb, audio, num_frames, label = labels, train = False)

	values, indices = tf.nn.top_k(predict, k=k)


	variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
	variables_to_restore = variable_averages.variables_to_restore()

	saver = tf.train.Saver(variables_to_restore)
	config = tf.ConfigProto(allow_soft_placement=True)
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess = tf.Session(config=config)
	sess.run(init)
	tf.train.start_queue_runners(sess=sess)

	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	saver.restore(sess, ckpt.model_checkpoint_path)

	example = 0
	
	with open("prediction.csv", "w") as f:
		f.write("VideoId,LabelConfidencePairs\n")
		while True:
			video_id_value, predict_value, indices_value = sess.run([video_id, values, indices])
			
			for i in xrange(0, len(video_id_value)): 
				f.write(str(video_id_value[i]) + ",")
				f.write(str(indices_value[i][0]) + " " + str(predict_value[i][0]))
				for j in xrange(0, k - 1):
					f.write(" " + str(indices_value[i][j+1]) + " " + str(predict_value[i][j+1]))
				f.write("\n")

			example += 1
			if example % 10000 is 0:
				print "{} examples are processed".format(example)

		

	