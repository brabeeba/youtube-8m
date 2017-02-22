import model
import tensorflow as tf
import sys

thismodule = sys.modules[__name__]

flags = tf.app.flags
FLAGS = flags.FLAGS

def inference(rgb, audio, num_frames, label = None):
	current_infer = getattr(thismodule, "inference{}".format(FLAGS.current_model))
	return current_infer(rgb, audio, num_frames, label = label)

def inference0(rgb, audio, num_frames, label = None):
	print "True"

