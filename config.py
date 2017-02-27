import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

server = True
current_model = 1

flags.DEFINE_bool('server', server, "A boolean to indicate if the training is at server or local")
flags.DEFINE_bool('new_model', False, "A boolean to indicate if this is a new model")

if server:
	flags.DEFINE_string('train_dir', "/n/coxfs01/brabeeba/youtube-8m/train_dir", "directory for training log and model")
	flags.DEFINE_string('data_video_dir', "/n/coxfs01/brabeeba/youtube-8m/data/video", "directory for training data at video level")
	flags.DEFINE_string('data_frame_dir', "/n/coxfs01/brabeeba/youtube-8m/data/frame", "directory for training data at frame level")
else:
	flags.DEFINE_string('train_dir', "./train_dir", "directory for training log and model")
	flags.DEFINE_string('data_video_dir', "./data/video", "directory for training data at video level")
	flags.DEFINE_string('data_frame_dir', "./data/frame", "directory for training data at frame level")
	

flags.DEFINE_integer('num_reader', 8, "Number of reader in the input pipeline")


flags.DEFINE_integer('max_steps', int(1e7), "maximum training step")
flags.DEFINE_integer('batch_size', 32, "batch size")

flags.DEFINE_integer('rgb_size', 1024, "Feature size of rgb")
flags.DEFINE_integer('audio_size', 128, "feature size for audio")
flags.DEFINE_integer('time_size', 300, "time step size for each sample")
flags.DEFINE_integer('num_class', 4716, "Number of class")

flags.DEFINE_float('lr', 0.001, "learning rate")
flags.DEFINE_integer('decay_steps', 300, "Number of step until decay")
flags.DEFINE_float('learning_decay_rate', 0.9999,"Decay rate for learning")

flags.DEFINE_float('forget_gate_init', 0.5, "Initialization value for forget gate in LSTM")

flags.DEFINE_float('beta1', 0.9, "Decay coefficient 1 for Adam")
flags.DEFINE_float('beta2', 0.999, "Decay coefficent 2 for Adam")
flags.DEFINE_float('moving_average_decay', 0.9999, "Decay coefficient for moving average")
flags.DEFINE_float('max_grad_norm', 5.0, "Maximum gradient norm")

flags.DEFINE_integer('current_model', current_model, "Current Model in Used")

if current_model == 0:
	flags.DEFINE_integer('rnn_rgb_hidden', 256, "Hidden Layer in rgn rnn")
	flags.DEFINE_integer('rnn_audio_hidden', 64, "Hidden Layer in audio rnn")

	flags.DEFINE_integer('rnn_rgb_layer', 1, "number of layer for rgb")
	flags.DEFINE_integer('rnn_audio_layer', 1, "number of layer for audio")

	flags.DEFINE_integer('final_hidden', 512, "hidden unit in last fc layer")
elif current_model == 1:
	flags.DEFINE_integer('rnn_rgb_hidden', 512, "Hidden Layer in rgn rnn")
	flags.DEFINE_integer('rnn_audio_hidden', 128, "Hidden Layer in audio rnn")

	flags.DEFINE_integer('rnn_rgb_layer', 1, "number of layer for rgb")
	flags.DEFINE_integer('rnn_audio_layer', 1, "number of layer for audio")

	flags.DEFINE_integer('final_hidden', 1024, "hidden unit in last fc layer")
	



