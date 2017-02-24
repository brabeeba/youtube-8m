# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils
import os
import sys
from tensorflow import logging

flags = tf.app.flags
FLAGS = flags.FLAGS

def resize_axis(tensor, axis, new_size, fill_value=0):
	"""Truncates or pads a tensor to new_size on on a given axis.

	Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
	size increases, the padding will be performed at the end, using fill_value.

	Args:
		tensor: The tensor to be resized.
		axis: An integer representing the dimension to be sliced.
		new_size: An integer or 0d tensor representing the new value for
			tensor.shape[axis].
		fill_value: Value to use to fill any new entries in the tensor. Will be
			cast to the type of tensor.

	Returns:
		The resized tensor.
	"""
	tensor = tf.convert_to_tensor(tensor)
	shape = tf.unstack(tf.shape(tensor))

	pad_shape = shape[:]
	pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

	shape[axis] = tf.minimum(shape[axis], new_size)
	shape = tf.stack(shape)

	resized = tf.concat([
			tf.slice(tensor, tf.zeros_like(shape), shape),
			tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
	], axis)

	# Update shape.
	new_shape = tensor.get_shape().as_list()  # A copy is being made.
	new_shape[axis] = new_size
	resized.set_shape(new_shape)
	return resized

class BaseReader(object):
	"""Inherit from this class when implementing new readers."""

	def prepare_reader(self, unused_filename_queue):
		"""Create a thread for generating prediction and label tensors."""
		raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
	"""Reads TFRecords of pre-aggregated Examples.

	The TFRecords must contain Examples with a sparse int64 'labels' feature and
	a fixed length float32 feature, obtained from the features in 'feature_name'.
	The float features are assumed to be an average of dequantized values.
	"""

	def __init__(self, num_classes):
		"""Construct a YT8MAggregatedFeatureReader.

		Args:
			num_classes: a positive integer for the number of classes.
			feature_sizes: positive integer(s) for the feature dimensions as a list.
			feature_names: the feature name(s) in the tensorflow record as a list.
		"""
		self.num_classes = num_classes

	def prepare_reader(self, filename_queue,):
		"""Creates a single reader thread for pre-aggregated YouTube 8M Examples.

		Args:
			filename_queue: A tensorflow queue of filename locations.

		Returns:
			A tuple of video indexes, features, labels, and padding data.
		"""
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)

		# set the mapping from the fields to data types in the proto
		
		feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
									 "labels": tf.VarLenFeature(tf.int64),
									 "mean_rgb": tf.FixedLenFeature([FLAGS.rgb_size], tf.float32),
									 "mean_audio": tf.FixedLenFeature([FLAGS.audio_size], tf.float32)}

		features = tf.parse_single_example(serialized_example,
																			 features=feature_map)

		labels = tf.cast(
				tf.sparse_to_dense(features["labels"].values, (self.num_classes,), 1,
						validate_indices=False), tf.int32)


		return features["video_id"], features["mean_rgb"], features["mean_audio"], labels

class YT8MFrameFeatureReader(BaseReader):
	"""Reads TFRecords of SequenceExamples.

	The TFRecords must contain SequenceExamples with the sparse in64 'labels'
	context feature and a fixed length byte-quantized feature vector, obtained
	from the features in 'feature_names'. The quantized features will be mapped
	back into a range between min_quantized_value and max_quantized_value.
	"""

	def __init__(self,num_classes,max_frames):
		"""Construct a YT8MFrameFeatureReader.

		Args:
			num_classes: a positive integer for the number of classes.
			feature_sizes: positive integer(s) for the feature dimensions as a list.
			feature_names: the feature name(s) in the tensorflow record as a list.
			max_frames: the maximum number of frames to process.
		"""
		self.num_classes = num_classes
		self.max_frames = max_frames

	def prepare_reader(self, filename_queue, max_quantized_value=2, min_quantized_value=-2):
		"""Creates a single reader thread for YouTube8M SequenceExamples.

		Args:
			filename_queue: A tensorflow queue of filename locations.
			max_quantized_value: the maximum of the quantized value.
			min_quantized_value: the minimum of the quantized value.

		Returns:
			A tuple of video indexes, video features, labels, and padding data.
		"""
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)

		contexts, features = tf.parse_single_sequence_example(
				serialized_example,
				context_features={"video_id": tf.FixedLenFeature([], tf.string),
								   "labels": tf.VarLenFeature(tf.int64)},
				sequence_features={
						"rgb" : tf.FixedLenSequenceFeature([], dtype=tf.string),
						"audio": tf.FixedLenSequenceFeature([], dtype=tf.string)
				})

		# read ground truth labels
		labels = tf.cast(tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1, 
			validate_indices=False), tf.int32)

		rgb = tf.reshape(tf.cast(tf.decode_raw(features["rgb"], tf.uint8), tf.float32), [-1, FLAGS.rgb_size])
		audio = tf.reshape(tf.cast(tf.decode_raw(features["audio"], tf.uint8), tf.float32), [-1, FLAGS.audio_size])
		num_frames = tf.minimum(tf.shape(rgb)[0], self.max_frames)
		tf.assert_equal(tf.shape(rgb)[0], tf.shape(audio)[0])

		rgb = resize_axis(utils.Dequantize(rgb,max_quantized_value,min_quantized_value), 0, self.max_frames)
		audio = resize_axis(utils.Dequantize(audio,max_quantized_value,min_quantized_value), 0, self.max_frames)
		

		return contexts["video_id"], labels, rgb, audio, num_frames

def generate_batch(inputs, min_queue_examples, batch_size, train):
	
	return tf.train.shuffle_batch_join(inputs, batch_size=batch_size,  capacity = min_queue_examples + 3 * batch_size, min_after_dequeue = min_queue_examples, allow_smaller_final_batch = True)
	

def input(train):
	files = tf.gfile.Glob(os.path.join(FLAGS.data_frame_dir,"train*.tfrecord"))
	filename_queue = tf.train.string_input_producer(files, shuffle = True)
	reader = YT8MFrameFeatureReader(FLAGS.num_class, FLAGS.time_size)

	inputs = [ reader.prepare_reader(filename_queue) for x in xrange(0, FLAGS.num_reader)]

	NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)


	video_id_batch, label_batch, rgb_batch, audio_batch, num_frames_batch = generate_batch(inputs, min_queue_examples, FLAGS.batch_size, train)
	
	return video_id_batch, label_batch, rgb_batch, audio_batch, num_frames_batch

	