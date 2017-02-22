import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', "./train_dir", "directory for training log and model")
flags.DEFINE_string('data_video_dir', "./data/video", "directory for training data at video level")
flags.DEFINE_string('data_frame_dir', "./data/frame", "directory for training data at frame level")
flags.DEFINE_integer('num_reader', 8, "Number of reader in the input pipeline")


flags.DEFINE_integer('max_steps', int(1e7), "maximum training step")
flags.DEFINE_integer('batch_size', 4, "batch size")


flags.DEFINE_float('lr', 0.001, "learning rate")
flags.DEFINE_integer('decay_steps', 300, "Number of step until decay")
flags.DEFINE_float('learning_decay_rate', 0.9999,"Decay rate for learning")

flags.DEFINE_float('forget_gate_init', 0.5, "Initialization value for forget gate in LSTM")

flags.DEFINE_float('beta1', 0.9, "Decay coefficient 1 for Adam")
flags.DEFINE_float('beta2', 0.999, "Decay coefficent 2 for Adam")
flags.DEFINE_float('moving_average_decay', 0.9999, "Decay coefficient for moving average")
flags.DEFINE_float('max_grad_norm', 5.0, "Maximum gradient norm")

flags.DEFINE_integer('current_model', 0, "Current Model in Used")

if FLAGS.current_model == 0:
	flags.DEFINE_integer('rnn_hidden', 256, "Hidden Layer in LSTM")

